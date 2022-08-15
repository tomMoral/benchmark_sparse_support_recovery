##################################
# Config file to skip some tests
#
# For each test <test_name>, benchopt checks if a function <check_test_name>
# exists in this file. If yes, it is run with the same input as the test
# before it is run. One can use `pytest.skip` or `pytest.xfail` to skip or
# mark the test as xfail.
