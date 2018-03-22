import json

print('loading json file...')

with open('test-result.json') as data_file:
    data = json.load(data_file)

for i in xrange(0, 60864):
    print i
    data[i]['question_id'] = int(float(data[i]['question_id']))

dd = json.dump(data,open('final_results.json','w'))


