import turtle
import random
import time
screen = turtle.Screen()

turtlepower = []

turtle.tracer(0, 0)
for i in range(1000):
    t = turtle.Turtle()
    t.goto(random.random()*500, random.random()*1000)
    turtlepower.append(t)

for i in range(1000):
    turtle.stamp()

turtle.update()

time.sleep(3)