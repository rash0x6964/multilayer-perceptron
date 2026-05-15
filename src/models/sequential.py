

class Sequential:
    def __init__(self, layers=None):
        self._layers = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)


    
