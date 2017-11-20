
units_row = [3,100,100,100,2]
class MLP(Chain):
	def __init__(self):
		super(MLP, self).__init__(
			l1 = L.Linear(units_row[0], units_row[1]),
			l2 = L.Linear(units_row[1], units_row[2]),
			l3 = L.Linear(units_row[2], units_row[3]),
			l4 = L.Linear(units_row[3], units_row[4]),
		)
	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		h3 = F.relu(self.l3(h2))
		y = self.l4(h3)
		return y
