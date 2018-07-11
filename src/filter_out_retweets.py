# this script reads in data downloaded from trumptwitterarchive.com and does some basic cleaning

o = open("../data/trump_2014-curr_cleaned.csv")
n = open("../data/_trump_2014-curr_cleaned.csv", "w")
for line in o:
    if line != "\n":
        n.write(line)
n.close()
o.close()



raise SystemExit
file_dir = "../data/"
filename = "trump_2014-curr"

read_file = open(file_dir + filename + ".csv")
write_file = open(file_dir + filename + "_no_rts.csv", "w")

# iterate through files, ignoring retweets and quote-tweets
for line in read_file:
    if (line[0:2] != 'RT') and (line[0] != '“') and (line[0] != '"'):
        write_file.write(line)

read_file.close()
write_file.close()

"""
Data is cleaned further with these bash commands:

to get rid of repeating periods:
cat trump_2014-curr_no_rts.csv | tr -s '.' >| tmp1.csv

to get rid of repeating hyphens:
cat tmp1.csv | tr -s '-' >| tmp2.csv

to get rid of hyperlinks:
sed 's/http:\/\/.*/ /' < tmp2.csv >| tmp3.csv
sed 's/https:\/\/.*/ /' < tmp3.csv >| trump_2014-curr_cleaned.csv
"""

