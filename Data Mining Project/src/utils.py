import pandas as pd
import os

from datetime import datetime
from dateutil.relativedelta import relativedelta

def read_to_df(filename):
    return pd.read_csv(os.getcwd() + f"/../data/{filename}", delimiter=";")

def read_date(date):
    date = str(date)

    return datetime.strptime(f"{int(date[:2]) + 1900}/{int(date[2:4]) % 50}/{date[4:6]}", "%Y/%m/%d")

def calculate_age(birth_date):
    end_date = datetime(1999, 1, 1)

    return relativedelta(end_date, birth_date).years