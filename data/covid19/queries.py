import pandas as pd
import numpy as np


class Task:
    def __init__(self, start_time, n_blocks, task_type):
        self.start_time = start_time
        self.n_blocks = n_blocks
        self.type = task_type


def generate_one_day_tasks(start_time, num_queries):
    tasks = []
    num_tasks = np.abs(np.random.normal(10, 20, 1)).astype(int)+1
    # print(num_tasks)
    for i in range(num_tasks[0]):
        # nblocks = np.abs(np.random.normal(1, 5, 1)).astype(int)[0]+1
        query_type = np.random.randint(1, num_queries+1)
        nblocks = np.random.choice([1, 7, 14], 1, p=[0.4, 0.4, 0.2])[0]
        tasks.append(Task(start_time, nblocks, query_type))
    return tasks
#


class Query:
    def __init__(self, query_type, blocks_num):
        self.query_type = query_type
        self.blocks_num = blocks_num

    def get_start_time(self):
        # At the early stages of the pandemic users are not very interested
        return np.random.randint(1, self.blocks_num)
        # mu, sigma = 30, 100
        # return np.abs(np.random.normal(mu, sigma, 1)).astype(int)+1

    def get_end_time(self, start_time):
        end_time = start_time + np.random.randint(1, 30)
        if end_time > self.blocks_num:
            end_time = self.blocks_num
        return end_time
        # mu, sigma = 30, 200
        # return start_time + np.abs(np.random.normal(mu, sigma, 1)).astype(int)+1

    def get_n_blocks(self, period):
        nblocks = np.random.randint(1, period+1)[0]
        # if nblocks > 5:
        #     nblocks = 5
        # nblocks = np.choice([1, 5], 1, p=[0.5, 0.5])
        # mu, sigma = 1, 1
        # nblocks = np.abs(np.random.normal(mu, sigma, 1)).astype(int)[0]+1
        # if nblocks > 5:
        #     nblocks = 5
        return nblocks

    def get_period(self,):
        period = np.random.choice([1, 5, 10, 15], 1, p=[0.25, 0.25, 0.25, 0.25])
        # period = np.random.randint(1, 10)
        return period
        # mu, sigma = nblocks, 10
        # return np.abs(np.random.normal(mu, sigma, 1)).astype(int)+1

    def generate_tasks(self,):
        start_time = self.get_start_time()
        end_time = self.get_end_time(start_time)
        period = self.get_period()

        tasks = []
        for i in np.arange(start_time, end_time, period):
            tasks.append(Task(i, self.get_n_blocks(period), self.query_type))
        return tasks


def query1(df):
    return df.query("continent == 'Europe'")['new_cases'].mean().rename(columns={'new_cases': 'result'})
    # df.groupby(by=["continent"]).mean()['new_cases'].to_frame('result')

def query2(df):
    return df.query("continent == 'Africa'")['new_cases'].mean().rename(columns={'new_cases': 'result'})

def query3(df):
    return df.query("continent == 'South America'")['new_cases'].mean().rename(columns={'new_cases': 'result'})

def query4(df):
    return df.query("continent == 'North America'")['new_cases'].mean().rename(columns={'new_cases': 'result'})

def query5(df):
    return df.query("continent == 'Asia'")['new_cases'].mean().rename(columns={'new_cases': 'result'})