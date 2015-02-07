'''.............................
   Creation 2015 by Samuel Tenka
   .............................'''
''' reader/writer of CSV files'''

def get_row(file):
   return [eval(s) for s in file.readline().split(',') if s]
def put_row(file, row):
   file.write(','.join(str(v) for v in row) + '\n')
class CSV:
    def __init__(self):
        self.headings = []
        self.rows = []
    def read_from(self, filename):
        self.rows = []
        with open(filename) as f:
            self.headings = f.readline().split(',')
            for row in iter(lambda:get_row(f), []):
                self.rows.append(row)
    def write_to(self, filename):
        with open(filename, 'w') as f:
            f.write(','.join(self.headings) + '\n')
            for row in self.rows:
                put_row(f, row)


'''
## for testing:
c = CSV()
c.read_from('D1_train.csv')
print(len(c.rows))
c.write_to('copy.csv')
'''
