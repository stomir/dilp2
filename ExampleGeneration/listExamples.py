import sys
import fire
import os
head = "head("
tail = "tail("
close = ")."
empty = "llempty"

def stringify(l):
	return "ll"+''.join([str(x) for x in l]) if l !=[] else empty

def listpreds(l,p):
	if l!=[]:
		s =stringify(l)
		t =l[1:]
		p["facts"].update([head+s+","+l[0]+close,tail+s+","+stringify(t)+close])
		listpreds(t,p)

def main(problem,name):
	with open(problem, 'r') as file:
		remove_nl= [ x.replace("\n", "") for x in file]
		cleaned = list(filter(lambda a: not a=="", remove_nl))
		ele, li , neg, pos =  cleaned[:cleaned.index("lists:")], \
		cleaned[cleaned.index("lists:")+1:cleaned.index("negatives:")],\
		cleaned[cleaned.index("negatives:")+1:cleaned.index("positives:")],\
		cleaned[cleaned.index("positives:")+1:]
		elements, lists = dict(zip(range(1,len(ele)+1),ele)),\
		dict(zip(range(len(ele)+2,len(ele)+len(li)+3),[ x.split(",") for x in li]+[[]]))
		preds = {"facts":set(),"negative":set(),"positive":set()}
		preds["facts"].add("empty("+empty+","+empty+close)

		for l in lists.values(): listpreds(l,preds)
		getelement = lambda a: elements[a] if a <= len(elements) else stringify(lists[a])
		for l in neg:
			args = l.split(" ")
			ret = name+"("
			for a in args:
				ret+= getelement(int(a))+","
			ret=ret[:len(ret)-1]+")."
			preds["negative"].add(ret)

		for l in pos:
			args = l.split(" ")
			ret = name+"("
			for a in args:
				ret+= getelement(int(a))+","
			ret=ret[:len(ret)-1]+")."
			preds["positive"].add(ret)

		current =os.path.abspath(os.getcwd())
		example_loc = os.path.join(current, "../examples/", name)
		if not os.path.exists(example_loc):
			os.mkdir(example_loc)

		for x in preds.keys():
			with open(example_loc+"/%s.dilp" % x, 'w' if \
		     not os.path.exists(example_loc+"/%s.dilp") else 'x' ) as file:
		  		for i in preds[x]: file.write(i+"\n")

if __name__ == "__main__":
	fire.Fire(main)
