from agents.gfp import GFPAgent
from agents.fql import FQLAgent
from agents.rebrac import ReBRACAgent
from agents.iql import IQLAgent
from agents.ifql import IFQLAgent
from agents.scsfql import SCSFQLAgent
from agents.scsgfp import SCSGFPAgent

agents = dict(
    gfp=GFPAgent,
    scsgfp=SCSGFPAgent,
    fql=FQLAgent,
    scsfql=SCSFQLAgent,
    rebrac=ReBRACAgent,
    iql=IQLAgent,
    ifql=IFQLAgent,
)
