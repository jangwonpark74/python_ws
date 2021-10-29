import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--Output", help = "Show Output")
parser.add_argument("-x", "--Help", help ="Show Help")

args = parser.parse_args()

if args.Output:
    print("Displaying Output as %s " % args.Output)

if args.Help:
    print("Displaying Help as per %s" % args.Help)
