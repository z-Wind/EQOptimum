import numpy as np

class PSO:
    pBest = []
    pBestFitness = []
    gBest = None
    gBestFitness = np.inf
    
    
    # 初始化
    def __init__(self, fitness, bounds, swarmSize=100, w=0.5, wp=0.5, wg=0.5):
        '''
        v = w*v + wp*(pBest-x) + wg*(gBest-x)
        '''
        self.swarmSize = swarmSize
        self.fitness = fitness
        self.pNum = len(bounds)
        self.bounds = bounds
        self.w = w
        self.wp = wp
        self.wg = wg
        
        # 初始化粒子和速度
        self.init_particles()

        # 初始化 fitness
        self.pBest = np.zeros((self.swarmSize, self.pNum))
        self.pBestFitness = np.ones(self.swarmSize) * np.inf
        self.updateFitness()
    
    
    # 初始化粒子
    def init_particles(self):
        self.particles = np.zeros((self.swarmSize, self.pNum))        
        self.v = np.zeros((self.swarmSize, self.pNum))
        
        for i, b in enumerate(self.bounds):
            self.particles[:,i] = np.random.uniform(b[0], b[1], self.swarmSize)
            self.v[:,i] = np.random.uniform(-b[1]+b[0], b[1]-b[0], self.swarmSize)
            
            
    # 增加粒子
    def produce_particles(self, addSize):
        self.swarmSize += addSize
        ps = np.zeros((addSize, self.pNum))        
        vs = np.zeros((addSize, self.pNum))
        pbests = np.zeros((addSize, self.pNum))
        pBestFitness = np.ones(addSize) * np.inf
        
        for i, b in enumerate(self.bounds):
            ps[:,i] = np.random.uniform(b[0], b[1], addSize)
            vs[:,i] = np.random.uniform(-b[1]+b[0], b[1]-b[0], addSize)
            
        self.particles = np.concatenate((ps, self.particles))
        self.v = np.concatenate((vs, self.v))
        self.pBest = np.concatenate((pbests, self.pBest))
        self.pBestFitness = np.concatenate((pBestFitness, self.pBestFitness))
        
        
    # 取代粒子
    def replace_particles(self, replaceSize):
        # 重新排序
        index = np.argsort(self.pBestFitness)[::-1]
        self.particles = self.particles[index]
        self.v = self.v[index]
        self.pBest = self.pBest[index]
        self.pBestFitness = self.pBestFitness[index]
        
        ps = np.zeros((replaceSize, self.pNum))        
        vs = np.zeros((replaceSize, self.pNum))
        pbests = np.zeros((replaceSize, self.pNum))
        pBestFitness = np.ones(replaceSize) * np.inf
        
        for i, b in enumerate(self.bounds):
            ps[:,i] = np.random.uniform(b[0], b[1], replaceSize)
            vs[:,i] = np.random.uniform(-b[1]+b[0], b[1]-b[0], replaceSize)

        self.particles[:replaceSize] = ps
        self.v[:replaceSize] = vs
        self.pBest[:replaceSize] = pbests
        self.pBestFitness[:replaceSize] = pBestFitness
    
    
    # 更新 fitness
    def updateFitness(self):
        for i, p in enumerate(self.particles):
            fit = self.fitness(p)
            if fit < self.pBestFitness[i]: 
                self.pBest[i] = p
                self.pBestFitness[i] = fit

                if fit < self.gBestFitness:
                    self.gBest = p
                    self.gBestFitness = fit
        
        
    def run(self, threshold=0.01, maxiter=100, randIter=20):
        n = 0
        try:
            while self.gBestFitness > threshold and n < maxiter:
                # 更新粒子速度
                rp = np.random.rand()
                rg = np.random.rand()
                self.v = self.w*self.v + self.wp*rp*(self.pBest-self.particles) + self.wg*rg*(self.gBest-self.particles)
                
                # 更新粒子
                self.particles = self.particles + self.v
                for i, b in enumerate(self.bounds):
                    self.particles[:,i] = np.clip(self.particles[:,i], b[0], b[1])
                
                # 計算 fitness
                old = self.gBestFitness
                self.updateFitness()
                print(self.gBestFitness)
                
                if old - self.gBestFitness < threshold/10:
                    n += 1
                    if n % randIter == 0:
                        self.replace_particles(self.swarmSize//3*2)
                        print('replace, now Size: {}'.format(self.swarmSize))
                        self.produce_particles(self.swarmSize//10)
                        print('produce, now Size: {}'.format(self.swarmSize))                        
                else:
                    n = 0
                if n > maxiter:
                    break                
                print(n)
                
        except KeyboardInterrupt:
            print("Interrupt by user")
            
        return self.gBest, self.gBestFitness
        
        
if __name__ == "__main__":
    def fitness(x):
        return (x[0]-5)**2 + (x[1]-10)**2 + (x[2]+10)**2

    pso = PSO(fitness, [(-1e10,1e10), (-1e10,1e10), (-1e10,1e10)])
    print(pso.run(threshold=1e-6))