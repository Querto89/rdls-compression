#database.py
import sqlite3
import os
import pandas as pd
from openpyxl import load_workbook

DB_FILE = 'db/rdls_basic_methods_db.db'

def initialize():
    """Inicjalizuje bazę danych i zwraca połączenie oraz kursor."""
    db_exists = os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if not db_exists:
        cursor.execute('''
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            image TEXT NOT NULL,
            transformation TEXT NOT NULL,
            filter TEXT NOT NULL,
            noise TEXT NOT NULL,
            format TEXT NOT NULL
        );
        ''')

        cursor.execute('''
        CREATE TABLE noises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            params JSON
        );
        ''')

        cursor.execute('''
        CREATE TABLE filters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            params JSON
        );
        ''')

        cursor.execute('''
        CREATE TABLE measurments_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            H0 FLOAT NOT NULL,
            H1 FLOAT NOT NULL,
            H0_R FLOAT,
            H0_G FLOAT,
            H0_B FLOAT,
            psnr FLOAT,
            bit_perfect INTEGER,
            hash_equal INTEGER,
            format TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            compression_ratio FLOAT,
            bpp FLOAT
        );
        ''')

        conn.commit()
        print("Baza danych została utworzona.")
    else:
        print("Baza danych istnieje. Połączenie nawiązane.")

    return conn, cursor

def db_save_image(cursor,image,color_space_transformation,filter_name,noise_name, compression_format):
    cursor.execute("INSERT INTO images (image,transformation,filter,noise,format) VALUES (?, ?, ?, ?, ?)", (image,color_space_transformation,filter_name,noise_name,compression_format))

def db_save_noise(cursor,noise_name,noise_params):
    cursor.execute("INSERT INTO noises (name,params) VALUES (?, ?)", (noise_name,noise_params))

def db_save_filter(cursor,filter_name,filter_params):
    cursor.execute("INSERT INTO filters (name,params) VALUES (?, ?)", (filter_name,filter_params))

def db_save_measurments_results(cursor,H0,H1,H0_R,H0_G,H0_B,psnr,bit_perfect,hash_equal,compression_format,file_size_bytes,compression_ratio,bpp):
    cursor.execute("INSERT INTO measurments_results (H0,H1,H0_R,H0_G,H0_B,psnr,bit_perfect,hash_equal,format,file_size_bytes,compression_ratio,bpp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (H0,H1,H0_R,H0_G,H0_B,psnr,bit_perfect,hash_equal,compression_format,file_size_bytes,compression_ratio,bpp))