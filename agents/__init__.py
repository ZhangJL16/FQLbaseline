from agents.gfp import GFPAgent
from agents.fql import FQLAgent
from agents.rebrac import ReBRACAgent
from agents.iql import IQLAgent
from agents.ifql import IFQLAgent

agents = dict(
    gfp=GFPAgent,
    fql=FQLAgent,
    rebrac=ReBRACAgent,
    iql=IQLAgent,
    ifql=IFQLAgent,
)
