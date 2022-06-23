import clingo #type: ignore
import fire #type: ignore
import re
import os
import random
from typing import *

#To run for front of a list use the following command:
#   python genTask.py front/list.lp front '{"ele":4,"li":4,"nli":1}' front '["head","tail","empty"]'--models 0
#To run for grandparent of a list use the following command:
#   python genTask.py ancestor.lp grandparent '{"generations":2,"members":5}' grandparent '["parent"]'--models 0

# Note, that setting models to 0 results in the construction of all models
# There may be many models.

def main(aspfile: str, problem : str,  args : Dict[str,int],
         target : str, bk : List[str], models : int = 1,
         counter : int = 1):
    ctrl = clingo.Control(str(models))
    with open(problem+"/"+aspfile, 'r') as file:
        program = ''.join(list(file))
        (const,val) = zip(*args.items())
        ctrl.add("main",const,program)
        ctrl.ground([("main",[clingo.Number(x) for x in val])])
        bkcapture = "|".join(["("+x+".*)" for x in bk])
        targetcapture = "("+target+".*)"
        anticapture = "(anti"+target+".*)"
        current =os.path.abspath(os.getcwd())
        example_loc = os.path.join(current, problem+"/"+problem+"/")
        if not os.path.exists(example_loc):
            os.mkdir(example_loc)
        count=counter
        lastnl = lambda a:  a[-1][:len(a[-1])-1]
        with ctrl.solve(yield_=True) as res: #type: ignore #I don't actually know, possibly type hints for clingo are incorrect?
            for m in res:
                mod = [x+".\n"  for x in "{}".format(m).split(" ")]
                par = list(filter(lambda x: True if re.match(bkcapture,x) else False,mod))
                gpar = list(filter(lambda x: True if re.match(targetcapture,x) else False,mod))
                agpar = list(filter(lambda x: True if re.match(anticapture,x) else False,mod))
                agpar = [x[4:] for x in agpar]
                par[-1] = lastnl(par)
                gpar[-1] = lastnl(gpar)
                agpar[-1] = lastnl(agpar)
                preds = {"facts":par,"negative":agpar,"positive":gpar}
                world_loc = os.path.join(example_loc, "world_"+str(count))
                if not os.path.exists(world_loc):
                    os.mkdir(world_loc)
                for x in preds.keys():
                    with open(world_loc+"/%s.dilp" % x, 'w' if \
                    not os.path.exists(world_loc+"/%s.dilp") else 'x' ) as file:
                        for i in preds[x]: file.write(i)
                count=count+1

if __name__ == "__main__":
	fire.Fire(main)
