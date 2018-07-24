
units_row = [19,100,19,1]
class MLP(Chain):
	def __init__(self):
		super(MLP, self).__init__(
			l1 = L.Linear(units_row[0], units_row[1]),
			l2 = L.Linear(units_row[1], units_row[2]),
			l3 = L.Linear(units_row[2], units_row[3]),
		)
	def pred(self, x):
		h1 = F.maxout(self.l1(x),1)
		h2 = F.maxout(self.l2(h1),1)
		y = self.l3(h2)
		return y
