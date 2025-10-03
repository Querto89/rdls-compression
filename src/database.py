#database.py
import sqlite3
import os
import pandas as pd
from openpyxl import load_workbook

DB_FILE = 'db/rdls_db.db'

def initialize():
    """Inicjalizuje bazę danych i zwraca połączenie oraz kursor."""
    db_exists = os.path.exists(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if not db_exists:
        cursor.execute('''
        CREATE TABLE images (
            id INTEGER PRIMARY K,
            image TEXT NOT NULL,
            transformation TEXT NOT NULL
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
        CREATE TABLE entropy_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            H0 FLOAT NOT NULL,
            H1 FLOAT NOT NULL,
            H0_R FLOAT,
            H0_G FLOAT,
            H0_B FLOAT,
            psnr FLOAT,
            bit_perfect INTEGER,
            hash_equal INTEGER
        );
        ''')

        cursor.execute('''
        CREATE TABLE compression_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def db_save_image(cursor,id,image,color_space_transformation):
    cursor.execute("INSERT INTO images (id,images,transformation) VALUES (?, ?, ?)", (id,image,color_space_transformation))#Zapisanie wybranego obrazu do tablicy images