1. Run 01 was 1 run for ancestral admgs sample sizes 500, 1000, 2000, 4000 - for relcadilac, dcd, gfci
2. Run 02 will be for 4 runs for the same as point 1 - so that in total they make 5 runs - can use commit b4a60cfd21c12864070a8a93fb07e6bb39521382 as reference
3. Run 03 will be for ancestral admgs varying the number of nodes [5, 10, 15, 20, 30] - for relcadilac, dcd, gfci - can use commit 80a3c11c7c8103f8b00cc6c740584b8 as reference
4. Run 04 is not a separate run, I have just combined the data from point 1 and point 2 to make a 5 run set for ancestral admgs with sample sizes 500, 1000, 2000, 4000 for relcadilac, dcd, gfci
5. For the above graphs can use commit 00d1e5877176df94dee36ac21da393549a270a7f as reference (for code and everything)
6. Run 05 is just a single test of relcadilac to see how long 3000 nodes data takes to run
7. In Run 06 I am again doing a 5 node 3000 node test to see how the rewards returned by relcadilac are handled
8. In run 07 I am trying to properly test the impact of sample size on the bic score - and how it compares with the ground truth bic - does the difference between the ground truth bic and the predicted bic (perhaps percentage difference) decrease as the sample size increases. I am also capturing some data on the average rewards to see whether 16000 is sufficient or I need to increase it
9. In run 08 I am running the bow free admg model on the sachs data set - without thresholding
10. In run 09 I am running the same test as run 01, but with thresholding applied - but this is only for relcadilac since I have added the thresholding, and want to recalculate its metrics. In order to do a comparision we would have to merge the data -- can use commit aa1a554e7a5e1d7cca458782e50e76937af04cee for code reference - there was a mistake conducting this test - the thresholding was applied incorrectly - rather than applying it to just the magnitude, the full value was compared
11. In run 10, I am running the same test as run 09, but with the bug in the code corrected - in the thresholding function in the utils.py file - but I am running the loop just twice rather than 5 times since 5 times takes too long - can use commit f437c4b0a18112e5f87aa61d12eaa11a9a9ec6a1 for code reference
12. In run 11, I am doing num nodes variation [5, 10, 15, 20, 30] with thresholding - for relcadilac, dcd, gfci for bow-free graphs
13. In run 12 I am doing sample size variation [500, 1000, 2000, 4000 with thresholding - for relcadilac, dcd, gfci for bow-free graphs
