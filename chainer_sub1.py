for idx in range(self.size-1):
    exec("l%d = L.Linear(%d, %d)," % (idx, layer_sizes[idx], layer_sizes[idx+1]))