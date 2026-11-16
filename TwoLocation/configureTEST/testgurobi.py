import gurobipy as gp
print(gp.gurobi.version())
m = gp.Model()
x = m.addVars(3000)
m.addConstr(x.sum() <= 1)
m.optimize()