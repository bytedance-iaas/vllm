import unittest

class TestCaseBase(unittest.TestCase):
    """
    TestCaseBase provides some useful common methods and defaults.

    It includes common attributes like:
      Logging
      Test directory
    """

    @classmethod
    def configClass(cls):
        """
        Custom initialization.
        """
        #app.init_logging()

    def __init__(self, test_id: str, name: str = "", group: str = "", output_dir: str = "~", description:str = ""):
       # TODO: define status, 
       #     0 means initialized
       #     1 means executed
       self._status = 0
       self.test_id = test_id
       self.name = name
       self.group = group
       self.output_dir = output_dir

    def setStatus(self, status):
       self.status = status
       
    @property
    def status(self):
       return self._status
    
    @property
    def outputDir(self):
        # Lazily initialize the directory on request.
        if self._outputDir is None:
          #testname = _BAD_FILENAME_CHARS.sub(self.id(), '-')
          #self._outputDir = tempfile.mkdtemp(prefix=full_testname + '-').rstrip('/')
          return self._outputDir

    def setUp(self):
        # reset the _outputDir for each single test.
        self._outputDir = None

    def assertJsonObjectsEqual(self, jo1, jo2):
        # assertEqual for json objects
        if jo1 is jo2:
          pass
        self.assertTrue(jo1 is not None and jo2 is not None)
        try:
          self.assertEqual(jo1, jo2)
        except AssertionError:
          k1 = set(jo1.keys())
          k2 = set(jo2.keys())
          # k1 & k2 must have the same set of keys
          self.assertFalse(k1.symmetric_difference(k2))
          for k in k1:
            self.assertEqual(jo1[k], jo2[k])
          # failed to find the diff!
          raise

