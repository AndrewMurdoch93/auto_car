import pickle

class test():
    def __init__(self, par_dict):
        self.alpha = par_dict['alpha']
        self.beta = par_dict['beta']


par_dict = {'alpha':0, 'beta':1}
a = test(par_dict)

outfile=open('testClassParameters', 'wb')
pickle.dump(par_dict, outfile)
outfile.close()

infile=open('testClassParameters', 'rb')
new_par_dict = pickle.load(infile)
infile.close()

class test():
    def __init__(self, par_dict):
        self.alpha = par_dict['alpha']
        self.beta = par_dict['beta']
        self.charlie = par_dict['charlie']

new_par_dict['charlie'] = 2
b = test(new_par_dict)

pass



