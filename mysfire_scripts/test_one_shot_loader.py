from mysfire import OneShotLoader

ll = OneShotLoader(filepath="/home/davidchan/Projects/mysfire/mysfire_scripts/simple_data.tsv")
print(ll([["0", "1", "hello world"], ["1", "2", "hello world 2"]]))
