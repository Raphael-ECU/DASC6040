# Copyright 2021 Bloomberg L.P.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Edited 2022 Raphael Du Sablon
#DASC 6040 Final Project
#East Carolina University

def main():
    
    import time
    import clustering_driver
    import clusteringdpmeans_driver
    import embedding_driver
    

    embedding_driver.main()

    #exec(open("embedding_driver.py").read())
    
    run_time = []
    
    
    #run pseudo-dpmeans and FANATIC with different quantities of documents
    #could this be a loop? Probably. Am I a lazy ctr-c/p bum? Yes.
    datasize = 200
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))


    
    datasize = 500
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))


    
    datasize = 1000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))


    
    datasize = 1500
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))


    
    datasize = 2000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))

    
    datasize = 5000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))


    
    datasize = 10000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))
    
      
    
    datasize = 20000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))
    

    
    
    datasize = 30000
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))
    
    
    datasize = None
    t0 = time.time()
    clusteringdpmeans_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for pseudodpmeans '))
    run_time.append((time.time() - t0))
        
    t0 = time.time()
    clustering_driver.main(datasize)
    run_time.append(('Data set size ' + str(datasize)))
    run_time.append(('Clustering time for FANATIC '))
    run_time.append((time.time() - t0))
    
    
    with open('.\output\PerformanceResults.txt', 'w') as f:
        for item in run_time:
            
            f.write("%s\n" % item)
            f.write("\n")
    
if __name__ == "__main__":
    main()