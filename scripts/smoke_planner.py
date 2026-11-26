"""Day 0 smoke test: solve the unified-planning 'hello world' problem with pyperplan."""
from unified_planning.shortcuts import (
    BoolType, Fluent, InstantaneousAction, Object, OneshotPlanner, Problem, UserType, get_environment,
)

get_environment().credits_stream = None

Location = UserType("Location")
robot_at = Fluent("robot_at", BoolType(), l=Location)
connected = Fluent("connected", BoolType(), a=Location, b=Location)

move = InstantaneousAction("move", src=Location, dst=Location)
src, dst = move.parameters
move.add_precondition(connected(src, dst))
move.add_precondition(robot_at(src))
move.add_effect(robot_at(src), False)
move.add_effect(robot_at(dst), True)

problem = Problem("robot")
problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(connected, default_initial_value=False)
problem.add_action(move)
locs = [Object(f"l{i}", Location) for i in range(1, 4)]
problem.add_objects(locs)
problem.set_initial_value(robot_at(locs[0]), True)
problem.set_initial_value(connected(locs[0], locs[1]), True)
problem.set_initial_value(connected(locs[1], locs[2]), True)
problem.add_goal(robot_at(locs[2]))

with OneshotPlanner(name="pyperplan") as planner:
    result = planner.solve(problem)
print(result.status)
for action in result.plan.actions:
    print(action)
