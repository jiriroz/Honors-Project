"""Count the number of rows in each table and pickle"""

tables = """airport10135  airport11042  airport11982  airport13204  airport14524
airport10136  airport11049  airport11986  airport13230  airport14543
airport10140  airport11057  airport11995  airport13232  airport14570
airport10141  airport11066  airport11996  airport13241  airport14574
airport10146  airport11067  airport11997  airport13244  airport14576
airport10154  airport11076  airport12003  airport13256  airport14588
airport10155  airport11092  airport12007  airport13264  airport14633
airport10157  airport11097  airport12012  airport13277  airport14635
airport10158  airport11109  airport12016  airport13290  airport14674
airport10165  airport11111  airport12094  airport13296  airport14679
airport10170  airport11122  airport12129  airport13303  airport14683
airport10185  airport11140  airport12156  airport13342  airport14685
airport10208  airport11146  airport12173  airport13344  airport14689
airport10245  airport11150  airport12177  airport13360  airport14696
airport10257  airport11193  airport12191  airport13367  airport14698
airport10268  airport11203  airport12197  airport13377  airport14709
airport10279  airport11233  airport12206  airport13388  airport14711
airport10299  airport11252  airport12217  airport13422  airport14730
airport10333  airport11259  airport12250  airport13424  airport14747
airport10361  airport11267  airport12255  airport13433  airport14771
airport10372  airport11274  airport12264  airport13459  airport14783
airport10397  airport11278  airport12265  airport13476  airport14794
airport10408  airport11292  airport12266  airport13485  airport14802
airport10423  airport11298  airport12278  airport13486  airport14814
airport10431  airport11308  airport12280  airport13487  airport14828
airport10434  airport11315  airport12320  airport13495  airport14831
airport10466  airport11336  airport12323  airport13502  airport14842
airport10469  airport11337  airport12335  airport13541  airport14843
airport10529  airport11413  airport12339  airport13577  airport14869
airport10551  airport11415  airport12343  airport13795  airport14893
airport10561  airport11423  airport12363  airport13796  airport14905
airport10577  airport11433  airport12389  airport13830  airport14908
airport10581  airport11447  airport12391  airport13851  airport14952
airport10590  airport11471  airport12397  airport13871  airport14955
airport10599  airport11481  airport12402  airport13873  airport14960
airport10620  airport11495  airport12436  airport13891  airport14986
airport10627  airport11503  airport12441  airport13930  airport15008
airport10631  airport11525  airport12448  airport13931  airport15016
airport10643  airport11537  airport12451  airport13933  airport15023
airport10666  airport11540  airport12478  airport13964  airport15024
airport10685  airport11577  airport12511  airport13970  airport15027
airport10693  airport11587  airport12519  airport14006  airport15041
airport10713  airport11603  airport12523  airport14025  airport15048
airport10721  airport11612  airport12758  airport14027  airport15070
airport10728  airport11617  airport12819  airport14057  airport15096
airport10731  airport11618  airport12884  airport14082  airport15249
airport10732  airport11624  airport12888  airport14098  airport15295
airport10739  airport11630  airport12889  airport14100  airport15304
airport10747  airport11637  airport12891  airport14107  airport15323
airport10754  airport11638  airport12892  airport14108  airport15356
airport10779  airport11641  airport12896  airport14109  airport15370
airport10781  airport11648  airport12898  airport14113  airport15374
airport10785  airport11695  airport12915  airport14122  airport15376
airport10792  airport11697  airport12945  airport14150  airport15380
airport10800  airport11721  airport12951  airport14193  airport15389
airport10821  airport11726  airport12953  airport14222  airport15401
airport10849  airport11775  airport12954  airport14252  airport15411
airport10868  airport11778  airport12982  airport14254  airport15412
airport10874  airport11823  airport12992  airport14256  airport15497
airport10918  airport11865  airport13024  airport14262  airport15582
airport10926  airport11867  airport13029  airport14288  airport15607
airport10930  airport11884  airport13061  airport14307  airport15624
airport10980  airport11898  airport13076  airport14321  airport15841
airport10990  airport11905  airport13121  airport14457  airport15897
airport10994  airport11921  airport13127  airport14487  airport15919
airport11002  airport11953  airport13158  airport14489  airport15991
airport11003  airport11973  airport13184  airport14492  airport16218
airport11013  airport11977  airport13198  airport14512
airport11041  airport11980  airport13203  airport14520"""

if __name__ == "__main__":
    tables = "\n".join(tables.split("  ")).split("\n")

    import sqlite3
    conn = sqlite3.connect("data/test.db")
    sizes = dict()
    for table in tables:
        cursor = conn.execute("SELECT count(*) from {}".format(table))
        sizes[table] = cursor.fetchall()[0][0]

    import pickle
    with open("data/tablesTest.p", "wb") as f:
        pickle.dump(sizes, f)

