

# class MinimizingAggregations(Planner):
#     def __init__(self, max_aggregations_allowed):
#         self.max_aggregations_allowed = max_aggregations_allowed
        
#     def get_execution_plan(self, query_id, blocks, budget):

#         max_num_aggregations = min(self.max_aggregations_allowed, len(blocks))

#         plan = []
#         for i in range(
#             max_num_aggregations + 1
#         ):  # Prioritizing smallest number of aggregations
#             splits = get_splits(blocks, i)
#             for split in splits:
#                 # print("split", split)

#                 for x in split:
#                     x = (x[0], x[-1])
#                     # print("         x", x)

#                     if self.fetch_result(query_id, x) is not None or self.can_run(
#                         self.scheduler, x, budget
#                     ):
#                         plan += [R(query_id, x, budget)]

#                     else:
#                         plan = []
#                         break

#                 if plan:
#                     if len(plan) == 1:
#                         return plan[0]
#                     else:
#                         return A(plan)
#         return None

#     def can_run(self, scheduler, blocks, budget):
    # demand = {}
    # for block in range(blocks[0], blocks[-1] + 1):
    #     demand[block] = budget
    # return scheduler.can_run(demand)
