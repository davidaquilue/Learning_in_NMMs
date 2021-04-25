''' Collection of file management functions. Used to simplify main.py

Contains: '''
import os

def check_create_results_folder():
    '''If results dir is not in the working dir it creates it'''
    path = os.getcwd()
    dir_list = os.listdir(path)
    if not ('Results' in dir_list):
        os.makedirs(path + '/Results')
    results_dir = path + '/Results'
    return results_dir

def test_folder(results_dir):
    ''' Checks the number of the last test performed and creates a folder of the actual test'''
    list_tests = os.listdir(results_dir)
    if len(list_tests) == 0:
        new_test = '/00'
        test_dir = results_dir + new_test
    else:
        last_test = int(list_tests[-1])
        print('Actual test number: ' + str(last_test + 1))
        if last_test < 9:
            new_test = '/0' + str(last_test + 1)
        else:
            new_test = '/' + str(last_test + 1)
        
        test_dir = results_dir  + new_test

    os.makedirs(test_dir)
    return test_dir



