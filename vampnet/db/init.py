"""
initialize the vampnet database. 

creates an empty duckdb database with 
tables for datasets, audio files, control signals, and splits.
"""
import duckdb
import vampnet


if __name__ == "__main__":
    vampnet.db.init()