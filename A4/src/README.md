# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

## Course Facilitator Setup
In order for you to successfully run each part of this assignment using the course facilitator A4 repo, you will need to change the following items
- Update `run.sh` test and test_local to point to  `./solution/test_outputs.txt` instead of `./submission/test_outputs.txt`. Since the submission directory is read-only, the command will run since opening the file requires write permissions as well.
- For test `1g-0-hidden` on line 668 in `grader.py` change the directory of `TEST_OUTPUT_FILE` to `./submission/test_outputs.txt` for the same reasons as above.
- To see what the expected `test_outputs.txt` file should be navigate to `alt-src` and there will be a `test-outputs.txt` file created from a successful trained model
