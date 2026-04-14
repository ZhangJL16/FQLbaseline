import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from utils.datasets import GPU_dataset_sample
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ConditionalGaussianSource, Value


class SCSFQLAgent(flax.struct.PyTreeNode):
    """State-conditioned source variant of FQL."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def condition_source_actions(self, observations, base_noises, grad_params=None):
        """Map standard Gaussian noises to a state-conditioned source."""
        means, log_stds, kappa = self.network.select("source")(
            observations, params=grad_params
        )
        stds = jnp.exp(log_stds)
        actions = means + stds * base_noises
        return actions, means, stds, log_stds, kappa

    def source_align_loss(self, source_means, target_actions):
        """Align source means with target actions by direction and distance."""
        source_norm = jnp.linalg.norm(source_means, axis=-1, keepdims=True)
        target_norm = jnp.linalg.norm(target_actions, axis=-1, keepdims=True)
        source_dir = source_means / jnp.maximum(source_norm, 1e-6)
        target_dir = target_actions / jnp.maximum(target_norm, 1e-6)
        cosine = jnp.sum(source_dir * target_dir, axis=-1)
        action_mag = jnp.squeeze(target_norm, axis=-1)
        direction_weight = action_mag / (action_mag + self.config["align_norm_c"])
        direction_loss = (
            self.config["align_direction_alpha"] * direction_weight * (1.0 - cosine)
        )
        l2_loss = self.config["align_l2_beta"] * jnp.mean(
            (source_means - target_actions) ** 2, axis=-1
        )
        align_loss = jnp.mean(direction_loss + l2_loss)
        return align_loss, {
            "align_cosine": jnp.mean(cosine),
            "align_direction_weight": jnp.mean(direction_weight),
            "align_l2": jnp.mean(l2_loss),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )
        agg_fn = getattr(jnp, self.config["q_agg"])
        next_q = agg_fn(next_qs, axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=grad_params
        )
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss with a state-conditioned source."""
        batch_size, action_dim = batch["actions"].shape
        rng, source_rng, t_rng, teacher_rng = jax.random.split(rng, 4)

        source_noises = jax.random.normal(source_rng, (batch_size, action_dim))
        x_0, source_means, source_stds, source_log_stds, source_kappa = self.condition_source_actions(
            batch["observations"], source_noises, grad_params=grad_params
        )
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        var_reg = 0.5 * (
            jnp.exp(2.0 * source_log_stds) - 1.0 - 2.0 * source_log_stds
        )
        var_reg_loss = jnp.mean(var_reg)
        align_loss, align_info = self.source_align_loss(source_means, x_1)

        teacher_noises = jax.random.normal(teacher_rng, (batch_size, action_dim))
        teacher_source_actions, _, _, _, _ = self.condition_source_actions(
            batch["observations"], teacher_noises
        )
        target_flow_actions = self.compute_flow_actions(
            batch["observations"], source_actions=teacher_source_actions
        )
        actor_actions = self.network.select("actor_onestep_flow")(
            batch["observations"], teacher_noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select("critic")(batch["observations"], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = (
            bc_flow_loss
            + self.config["lambda_var"] * var_reg_loss
            + self.config["lambda_align"] * align_loss
            + self.config["alpha"] * distill_loss
            + q_loss
        )

        actions = self.sample_actions(batch["observations"], seed=rng)
        mse = jnp.mean((actions - batch["actions"]) ** 2)

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "var_reg_loss": var_reg_loss,
            "align_loss": align_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q": q.mean(),
            "mse": mse,
            "source_mean_abs": jnp.abs(source_means).mean(),
            "source_std_mean": source_stds.mean(),
            "source_kappa": jnp.mean(source_kappa),
            **align_info,
        }

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
    def multi_sample_and_update(
        self,
        n_steps_fused,
        dataset,
        dataset_size,
        batch_size,
    ):
        _agent = self
        for _ in range(n_steps_fused):
            batch, rng = GPU_dataset_sample(
                _agent.rng, dataset, batch_size, dataset_size
            )
            rng, loss_rng = jax.random.split(rng)

            def loss_fn(grad_params):
                return _agent.total_loss(batch, grad_params, rng=loss_rng)

            new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
            _agent.target_update(new_network, "critic")
            _agent = _agent.replace(network=new_network, rng=rng)
        return _agent, info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["action_dim"],
            ),
        )
        actions = self.network.select("actor_onestep_flow")(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        source_actions,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_bc_flow_encoder")(observations)
        actions = source_actions
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample source actions and integrate the BC flow teacher."""
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["action_dim"],
            ),
        )
        source_actions, _, _, _, _ = self.condition_source_actions(observations, noises)
        return self.compute_flow_actions(observations, source_actions)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()
            encoders["actor_onestep_flow"] = encoder_module()
            encoders["source"] = encoder_module()

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_bc_flow"),
            need_time=True,
            encode_time_dim=None,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_onestep_flow"),
            need_time=False,
        )
        source_def = ConditionalGaussianSource(
            hidden_dims=config["source_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["source_layer_norm"],
            log_std_min=config["source_log_std_min"],
            log_std_max=config["source_log_std_max"],
            encoder=encoders.get("source"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
            source=(source_def, (ex_observations,)),
        )
        if encoders.get("actor_bc_flow") is not None:
            network_info["actor_bc_flow_encoder"] = (
                encoders.get("actor_bc_flow"),
                (ex_observations,),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
