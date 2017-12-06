'''
    exceptions.py
    @author Joey Franc
    
    This file is for defining exception classes to better
    communicate errors that occur in the pipeline.

'''



class MissingClassifierError(FileNotFoundError):
    
    def __init__(self, file_name):
        super().__init__(
                'Classifier file: '
                + file_name
                + '.py does not exist')


class MissingFeatureError(FileNotFoundError):
    
    def __init__(self, file_name):
        super().__init__(
                'Feature file: '
                + file_name
                + '.py does not exist')


class InvalidTensorError(ValueError):
    
    def __init__(self, invalid):
        super().__init__(
            "Tensor shape: "
            + str(invalid.shape)
            + ' is not of dimensions 2 or 3.')
        

class InconsistentTensorError(ValueError):
    
    def __init__(self, destination, invalid, feature):
        super().__init__(
                'Feature:  '
                + feature
                + 'with shape '
                + str(invalid.shape)
                + ' cannot be concatenated to a tensor of shape '
                + str(destination.shape[:-1])[:-1] + ', X)')


class InvalidClassifierError(ValueError):
    
    def __init__(self, classifier):
        super.__init__(
            'Classifier file: '
            + classifier
            + '.py does not exist')
