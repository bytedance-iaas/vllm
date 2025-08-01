from test_case import TestCaseBase

class Condition:
    def __init__(self, name, check_function):
        self.name = name
        self.check_function = check_function

    def isSatisfied(self, *args, **kwargs):
        return self.check_function(*args, **kwargs)

# Example usage
def checkValue(temp):
    return temp > 25

def check_humidity(humidity): return humidity < 50

class ParameterAdjust():
    def __init__(self, name: str, delta: float = 1.0, increase: bool = True, priority: int = 0):
        self.name = name
        self.delta = delta
        self.increase = increase
        self.priority = 0

    def adjust(self):
        if self.increase:
            self.value += self.delta
        else:
            self.value -= self.delta

class VllmTestCase(TestCaseBase):
    def __init__(self, test_id: str, name: str = "", group: str = "", output_dir: str = "~", description: str=""):
        super().__init__(test_id = test_id, name = name, group = group, output_dir = output_dir)
        self.adjust_parameters = []
        self.expected_conditions = []

    # The parameter-related functions are for adjusting 
    # parameters dynamically
    def addAdjustParameter(self, param: ParameterAdjust):
        if param is None:
            print("addJustParameter: parameter cannot be null")
            return

        # TODO: Add parmater validation logic
        self.adjust_parameters.append(param)

    def findAdjustParameter(self, name: str = None):
        for p in self.adjust_parameters:
            if p.name == name:
                return p
        return None

    def adjustParameters(self, name: str = None):
        if name is not None:
            # find the parameter
            p = self.findAdjustParameter(name)
            if p is not None:
                p.adjust()
        else:
          print("Not implemented")
          # TODO
          # Sort based upon the priority of each parameter adjustment

