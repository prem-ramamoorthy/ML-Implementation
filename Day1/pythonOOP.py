class Student: # Class definition
    college = "VIT"   # class attribute

    def __init__(self, name): # Constructor
        self.name = name  # instance attribute and public attribute
        self.__private_attr = "This is private" # private attribute allowed only within class
        self._protected_attr = "This is protected" # protected attribute allowed within class and subclasses
        
    @classmethod
    def Classmethod(cls): # Class Method
        print(cls) # prints the class reference
        print(cls.college)
        return "Hi!"

    def InstanceMethod(self): # Instance Method
        print(self) # prints the instance reference
        print(self.name)
        return "Hi!"
    
    @staticmethod
    def StaticMethod(): # Static Method
        print("Hello from Static Method") # No reference to class or instance just logic Implementation
        return "Hi!"
    
    def __del__(self): # Destructor
        print(f"Destructor called for {self.name}")

s1 = Student("Alice") # Instance / object with attribute name "Alice"

class A: pass # Empty class A
class B(A): pass # Class B inherits from class A (Single Inheritance)

class A: pass
class B(A): pass
class C(A): pass  # Multilevel Inheritance

class A: pass
class B(A): pass
class C(A): pass  # Hierarchical Inheritance

class A: pass
class B: pass
class C(A, B): pass # Multiple Inheritance

# Abstraction using Abstract Base Class (ABC)

from abc import ABC, abstractmethod

class AbstractStudent(ABC): # Abstract Base Class
    
    @abstractmethod
    def show(self): # Abstract Method can be declared only in Abstract Class
        pass

# AbstractStudent cannot be instantiated directly
class ConcreteStudent(AbstractStudent): # Concrete class inheriting from Abstract class
    def show(self): # Implementing the abstract method without which error will be raised
        print("Implementation of abstract method")
        
class student2(Student): # Inheriting from Student class
    def InstanceMethod(self): # Overriding Instance Method
        print("Overridden Instance Method")
        print(self._protected_attr) # Accessing protected attribute from parent class
        
    def __add__(self, other): # Operator Overloading  and this method is also called Dunder/magic method
        return self.name + " " + other.name
    
s2 = student2("Bob") # Instance of student2 with name "Bob"
s2.InstanceMethod() # Calling overridden method
print(s2 + s1) # Using overloaded + operator