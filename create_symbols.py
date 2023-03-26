import sys
import time
import unittest
import os
from optparse import OptionParser


def get_all_library_file(src_dir):
    # print(src_dir)
    library_files = []
    for parent, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename[-3:] == ".so":
                library_files.append(src_dir+"/"+filename)
    return library_files


def create_symbol_file(bin_dir, library_file):
    symbol_file = "symbols.sym"
    cmd = bin_dir + "/dump_syms " + library_file+ " >" +symbol_file
    # print(cmd)
    os.system(cmd)
    return symbol_file


def modify_symbol_file(dst_dir, symbol_file):
    cmd = "head -n1 "+symbol_file
    # print(cmd)
    result = os.popen(cmd).readlines()
    # print(result[0])
    content = result[0].split(" ")
    # print(content)
    library_name = content[4][0:-1]
    module_name = content[3]
    file_name = library_name+".sym"
    mv_dst_dir = dst_dir+"/"+library_name+"/"+module_name
    mv_dst = mv_dst_dir+"/"+file_name
    cmd = "mkdir -p "+mv_dst_dir
    # print(cmd)
    os.system(cmd)
    cmd = "mv "+symbol_file+" "+mv_dst
    # print(cmd)
    os.system(cmd)
    pass


def main():
    parser = OptionParser(
        usage="usage: %prog [options] <dump_syms binary> <symbol bin path> <dst symbol path>"
    )
    parser.add_option(
        "-b",
        "--bin_dir",
        action="store", type="string", dest="bin_dir",
    )
    parser.add_option(
        "-s",
        "--src_dir",
        action="store",
        type="string", 
        dest="src_dir",
    )
    parser.add_option(
        "-d",
        "--dst_dir",
        action="store", type="string", 
        dest="dst_dir",
    )
    (options, args) = parser.parse_args()  

    all_lib = get_all_library_file(options.src_dir)
    # print(all_lib)
    for lib in all_lib:
        # print(lib)
        symbol_file = create_symbol_file(options.bin_dir,lib)
        modify_symbol_file(options.dst_dir,symbol_file)
        # break
        


# run main if run directly
if __name__ == "__main__":
    main()
