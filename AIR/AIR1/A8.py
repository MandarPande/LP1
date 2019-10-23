class Move:
    def __init__(self,s=""):
        self.s=s
        self.g=0
        self.h=0
        self.f=0
        self.cost_of_move=0
        self.parent=None
        self.hist=[]
    

class Tiles:
    def __init__(self):
        self.target=""
        self.cost=0

    def h(self,s):
        misplaced=0
        for i in range(len(s)):
            if i<len(s)/2 and s[i]=='W':
                misplaced+=1
            elif i>=len(s)/2 and s[i]=='B':
                misplaced+=1
        return misplaced
    
    def check(self,s,i,step):
        if i+step>len(s)-1 or i+step<0:
            return False
        
        return True
    def print_list(self,l):
        for x in l:
            print("{} : {}".format(x.s,x.h))
    def calculate(self,s):
        #First calculate target
        for i in range(len(s)):
            if i<len(s)/2:
                self.target+="B"
            else:
                self.target+="W"
        
        print("Target : ",self.target)

        open=[]
        close=[]

        start=Move(s)
        start.h=self.h(s)
        start.g=1
        start.f=start.g+start.h
        start.parent=None
        start.cost_of_move=0
        start.hist.append(start)
        open.append(start)
        found=False
        ans=Move()
        while not found:
            while len(open)!=0 and not found:
                obj=Move()
                obj=open[0]
                del open[0]
                #print("Level {} : {} ".format(obj.g,obj.s))
                successors=[]
                for i in range(len(obj.s)):
                    for j in range(1,3):
                        temp=obj.s
                        if self.check(temp,i,j):
                            temp=list(temp)
                            temp[i],temp[i+j]=temp[i+j],temp[i]
                            temp=''.join(temp)
                            if(temp!=obj.s):
                                m=Move(temp)
                                m.h=self.h(m.s)
                                m.g=obj.g+1
                                m.f=m.g+m.h
                                m.parent=obj
                                m.cost_of_move=j
                                m.hist=obj.hist.copy()
                                m.hist.append(m)
                                successors.append(m)
                    

                for successor in successors:
                            #print("Current : ",temp)
                    if successor.s == obj.s:
                        continue
                    elif successor.s==self.target:
                        found=True
                        print("FOUND!!")
                        ans=successor
                        break
                    else:
                        flag = False
                        for move in open:
                            if(move.f<=successor.f and move.s==successor.s):
                                flag=True
                                break
                            elif(move.f>successor.f and move.s==successor.s):
                                open.remove(move)


                        if flag:
                            continue
                        for move in close:
                            if(move.f<=successor.f and move.s==successor.s):
                                flag=True
                                break
                            elif(move.f>successor.f and move.s==successor.s):
                                close.remove(move)

                        if flag:
                            continue
                        else:
                            #print("Appended : ",successor.s)
                            open.append(successor)
                    '''
                    print("OPEN LIST")
                    self.print_list(open)

                    print()
                    print("CLOSE LIST")
                    self.print_list(close)
                    '''            


                           
                close.append(obj)
                open.sort(key=lambda x: x.h)#WHY SORT?
                #self.print_list(open)

               

        
        for move in ans.hist:
            print("Value : {}\t g(x) : {}\th(x) :{} \t f(x) :{}\t Cost of move : {}".format(move.s,move.g,move.h,move.f,move.cost_of_move))
            self.cost+=move.cost_of_move
        '''
        move=ans
        while move != None:
            print("Value : {}\t g(x) : {}\th(x) :{} \t f(x) :{}\t Cost of move : {}".format(move.s,move.g,move.h,move.f,move.cost_of_move))
            self.cost+=move.cost_of_move
            move=move.parent
        '''
        print("Total cost : ",self.cost)

        

def is_valid_input(s):
    b=s.count("B",0,len(s))
    w=s.count("W",0,len(s))

    #print(b,w)
    return b==w 

if __name__=='__main__':
    t=Tiles()
    
    s=input("Enter string : ")
    while not is_valid_input(s):
        print("Number of B and W should be same")
        s=input("Enter string : ")
    t.calculate(s)  





