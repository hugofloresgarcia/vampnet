"""
initialize the vampnet database. 

creates an empty sqlite3 database with 
tables for datasets, audio files, control signals, and splits.
"""
import sqlite3
import vampnet


if __name__ == "__main__":
    vampnet.db.init()