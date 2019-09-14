import math

class Shape2D:
  def area(self):
    raise NotImplementedError()

class Square(Shape2D):
  def __init__(self, width):
    self.width = width
   
  def area(self):
    return self.width **2

class Disk(Shape2D):
  def __init__(self, radius):
    self.radius = radius
   
   def area(self):
     return math.pi * self.radius **2
 
 shapes = [Square(2), Disk(3)]
 
 ## Polymorphism
 print([s.area() for s in shapes])
 
 s = Shape2D()
 try:
  s.area()
 except NotImplementedError as e:
   print("NotImplementedError")
