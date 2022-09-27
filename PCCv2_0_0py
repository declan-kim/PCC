'''
PCC v 2.0.0
GCV v1.1.1 에서
1. 보행자신호 자동연장, 음성안내 보조장치, 보행 대기자 검출 장치의 영역을 각각 
   실제 영역에 맞게 구분을 두어 벡터 알고리즘의 실행 때에 각각의 장치의 요구하는
   출력 인터페이스에 맞추어 결과를 출력
2. Crawler 알고리즘에서 중심점을 찾는 것을 각 데이터의 최소, 최대값을 통해 구하고
    있었으나 이를 좀더 실제와 같은 신호로 개선
3. 
'''

import socket
import struct
import datetime
import math
import cv2
import numpy as np

class Lidar_2way_com:
    '''라이다를 윈도우 컴퓨터에 연결해서 연결속성을 1ch로 바꾸어야한다.
    추후 라이다와의 통신을 통해 개선할 수 있을것이다. - 우선순위 낮음'''
    targetIP = "224.0.0.5"      #라이다 멀티캐스트 주소
    targetPort = 5000           #라이다 멀티캐스트 포트
    target = ('',targetPort)

    bufferSize = 1024           #라이다의 데이터 길이가 980이기 때문에 이보다는 커야한다.
    UDPtargetSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM,proto=socket.IPPROTO_UDP)
    UDPtargetSocket.bind(target)
    UDPtargetSocket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #해당 주소를 사용하고 있었다 하더라도 그대로 재사용
    '''아래의 두 setsockopt()함수는 같은 기능을 담당하고 있으나 mreq를 통해 초기화한 
    부분은 윈도우에서의 통신을 위한 것이고 아래의 두개의 주소를 더하여 어뎁터의 주소와 
    멀티캐스트 주소를 정의한 것은 우분투-라즈비안 운영체제에서의 주소를 설정한 것이다.'''
    mreq = struct.pack("4sl",socket.inet_aton(targetIP),socket.INADDR_ANY)
    UDPtargetSocket.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,mreq)
    #for ubuntu -raspbian, eth0 internet interface
    #UDPtargetSocket.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,socket.inet_aton("224.0.0.5")+socket.inet_aton("192.168.123.100"))
    nomal_x = []
    nomal_y = []
    index = 3
    def data_nomalization_x(self,v):
        self.nomal_x.append(v)
        if self.nomal_x.__len__() >self.index :
            self.nomal_x.reverse()
            self.nomal_x.pop()
            self.nomal_x.reverse()
        else :
            return v
        all = 0
        for v_x in self.nomal_x:
            all = all +v_x
        nomal = all/self.index
        return nomal
    def data_nomalization_y(self,v):
        self.nomal_y.append(v)
        if self.nomal_y.__len__() >self.index :
            self.nomal_y.reverse()
            self.nomal_y.pop()
            self.nomal_y.reverse()
        else :
            return v
        all = 0
        for v_y in self.nomal_y:
            all = all +v_y
        nomal = all/self.index
        return nomal
    def getLidarData(self):
        msg, addr = self.UDPtargetSocket.recvfrom(self.bufferSize)
        header = msg[0:8]               #헤더
        msg = msg[7:msg.__len__()-2]    #체크섬값을 제외한 데이타 부분
        hex_msg = msg.hex()             #바이트 정보를 헥사코드로 바꾼다
        sensor_v = []                   #센서의 480개 데이터가 담기는 변수
        count = 1                       #센서값의 실제 위치를 판독하기 위한 카운트, 1 == 0.25도
        for iter in range(0,hex_msg.__len__(),4):
            '''라이다에서 신호를 m와 cm를 각각 한개의 헥사코드로 방송을 한다. 여기서
            한개의 헥사코드는 2개의 포인트를 먹는다. 따라서 한점의 실제 값을 얻기위해서는
            4개의 포인트를 해석 함으로서 신호를 해석할 수 있다.'''
            value =int(hex_msg[iter:iter+2],16)+int(hex_msg[iter+2:iter+4],16)/100 # m 단위
            x= value*math.cos(0.25*count/180*math.pi) #삼각비
            y= value*math.sin(0.25*count/180*math.pi) #삼각비
            x = self.data_nomalization_x(x)
            y = self.data_nomalization_y(y)
            sensor_v.append([x,y])
            count = count +1
        return sensor_v
        
class ObjectDetector_Crawler:
    old_crawler_obj_list = []   # 현재프레임
    cur_crawler_obj_list = []   # 1프레임 전
    old_x, old_y = -1, -1       # 직전 그리드 값 혹은 센서값
    crawl_start = -1
    data_counter = 0            #그리드 값, 센서값 인덱스
    crawler_count = 1           #지렁이 알고리즘으로 발견한 객체번호
    threshold = 17              #같은 객체라고 인식하기위한 거리 cm
    min_x,max_x,min_y,max_y = -1,-1,-1,-1
    def crawler(self,x,y, shut):
        '''직선의 방정식을 통한 현재 센서값과 직전의 센서값 사이의 거리 측정
        이를 통해 각각의 점이 원으로 얼머나 멀리떨어져있는가를 알 수 있음'''
        line = (self.old_x - x)*(self.old_x - x)+(self.old_y -y)*(self.old_y -y)
        dis = math.sqrt(line)
        if self.crawl_start == -1:                  #초기화 상태
            self.crawl_start = self.data_counter    #시작점 정보 저장
            self.min_x = x
            self.max_x = x
            self.min_y = y
            self.max_y = y
        elif dis <= self.threshold and not shut:                 # 지정된 거리이하면 같은 객체로 인식
            if self.min_x == -1:
                '''초기값 입력'''
                self.min_x = x
                self.max_x = x
                self.min_y = y
                self.max_y = y
            else :
                '''매 연산마다 최소값과 최대값 교환'''
                self.min_x = min(self.min_x, x)
                self.max_x = max(self.max_x, x)
                self.min_y = min(self.min_y, y)
                self.max_y = max(self.max_y, y)
        else :
            '''이 경우는 문턱값보다 거리가 길다는 의미로 크로울러 알고리즘에서는
            서로 다른 객체로 받아들인다. 따라서 이 값을 객체로 저장한다. '''
            if abs(self.crawl_start-self.data_counter)>3: 
                self.cur_crawler_obj_list.append([self.crawl_start,self.data_counter-1,\
                        self.min_x,self.max_x,self.min_y,self.max_y,self.crawler_count,\
                        0,[0.0,0.0],[0.0,0.0],[0.0,0.0]])
            '''사용하는 변수 초기화, 다음 연산을 대비'''
            self.crawl_start = -1
            self.min_x = -1
            self.crawler_count = self.crawler_count +1
        self.data_counter = self.data_counter +1    #크로울러 알고리즘에서 인식하는 센서값의 위치
        self.old_x = x                              #직전 프레임의 x
        self.old_y = y                              #직전 프레임의 y
    def init_v(self):
        '''다음 연산에서 필요한 값 초기화'''
        self.cur_crawler_obj_list = []
        self.old_x, self.old_y = -1, -1       # 직전 그리드 값 혹은 센서값
        self.crawl_start = -1
        self.data_counter = 0            #그리드 값, 센서값 인덱스
        self.crawler_count = 1           #지렁이 알고리즘으로 발견한 객체번호
        self.min_x,self.max_x,self.min_y,self.max_y = -1,-1,-1,-1

class GCV_Controller:
    real_x = -1         # 실측 라이다 범위, 가로
    real_y = -1         # 실측 라이다 범위, 세로

    crawl = ObjectDetector_Crawler()
    crawl_list = []
    def makeMAT(self, src, value):
        prefix = int(self.real_y*math.sin(30/180*math.pi)/math.cos(30/180*math.pi))
        for v in value:
            cv2.rectangle(
                src,
                (int(v[0]*100+prefix), int(v[1]*100)),
                (int(v[0]*100+1+prefix), int(v[1]*100+1)),
                (0,0,0),
                -1,
                cv2.LINE_AA
            )
    def __init__(self, x, y):
        self.real_x = x
        self.real_y = y
    def init(self):
        self.crawl_list = []        #매 프레임마다 초기화
    def compute(self, value):
        prefix = int(self.real_y*math.sin(30/180*math.pi)/math.cos(30/180*math.pi))
        for v in value:
            x , y = v[0]*100+prefix, v[1]*100
            self.crawl.crawler(v[0]*100+prefix, v[1]*100,False)
        self.crawl.crawler(x,y,True)
        count = 0
        for cur_obj in self.crawl.cur_crawler_obj_list:
            try:
                cur_obj_x = ((cur_obj[2]+cur_obj[3])/2)
                cur_obj_y = ((cur_obj[4]+cur_obj[5])/2)
                if self.crawl.old_crawler_obj_list.__len__() >0:
                    for old_obj in self.crawl.old_crawler_obj_list:
                        old_obj_x = ((old_obj[2]+old_obj[3])/2)
                        old_obj_y = ((old_obj[4]+old_obj[5])/2)
                        if abs(math.sqrt((cur_obj_x-old_obj_x)*(cur_obj_x-old_obj_x) + (cur_obj_y-old_obj_y)*(cur_obj_y-old_obj_y)))<17:
                            self.crawl.cur_crawler_obj_list[count][6] = old_obj[6] #obj id
                            self.crawl.cur_crawler_obj_list[count][7] = old_obj[7] + 1 #counts
                            self.crawl.cur_crawler_obj_list[count][8] = old_obj[8] # 3
                            self.crawl.cur_crawler_obj_list[count][9] = old_obj[9] # 2
                            self.crawl.cur_crawler_obj_list[count][10] = old_obj[10] # 1
                            if self.crawl.cur_crawler_obj_list[count][7] >3:
                                self.crawl.cur_crawler_obj_list[count][7] = 3
                                calc_x = abs(self.crawl.cur_crawler_obj_list[count][8][0] -cur_obj_x)
                                if calc_x <=0.3:
                                    calc_x = 0.3
                                elif calc_x <= 0.5:
                                    calc_x = 0.5
                                elif calc_x <= 1:
                                    calc_x = 1
                                calc_y = abs(self.crawl.cur_crawler_obj_list[count][8][1] - cur_obj_y)
                                t_value = 10 # 떨림 보정수치
                                self.crawl.cur_crawler_obj_list[count][8] = old_obj[9]
                                self.crawl.cur_crawler_obj_list[count][9] = old_obj[10]
                                self.crawl.cur_crawler_obj_list[count][10] = [cur_obj_x,cur_obj_y]
                                if (old_obj[8][0]+t_value>cur_obj_x and  old_obj[8][0]-t_value<cur_obj_x) and (old_obj[8][1]+t_value>cur_obj_y and  old_obj[8][1]-t_value<cur_obj_y):
                                    self.crawl_list.append([[cur_obj[2], cur_obj[3],cur_obj[4], cur_obj[5]],[int(old_obj[8][0]),int(old_obj[8][1]),int(cur_obj_x), int(cur_obj_y)],(255,0,0)])
                                elif abs(calc_y/calc_x) >1.5:
                                    self.crawl_list.append([[cur_obj[2], cur_obj[3],cur_obj[4], cur_obj[5]],[int(old_obj[8][0]),int(old_obj[8][1]),int(cur_obj_x), int(cur_obj_y)],(0,255,0)])
                                elif abs(calc_y/calc_x) <=1.5:
                                    self.crawl_list.append([[cur_obj[2], cur_obj[3],cur_obj[4], cur_obj[5]],[int(old_obj[8][0]),int(old_obj[8][1]),int(cur_obj_x), int(cur_obj_y)],(0,0,255)])
                count = count +1
            except KeyboardInterrupt:
                exit()
            except ZeroDivisionError:
                print("zero Division")
                exit()
        self.crawl.old_crawler_obj_list = self.crawl.cur_crawler_obj_list
        self.crawl.init_v()
        return self.crawl_list

if __name__ == "__main__":
    real_x = 1100
    real_y = 800
    con = GCV_Controller(real_x, real_y)
    sensor = Lidar_2way_com()
    #sensor = Lidar()
    oldtime = datetime.datetime.now()
    cv2.namedWindow('grid')
    img = np.ones((real_y, real_x+int(real_y*math.sin(30/180*math.pi)/math.cos(30/180*math.pi)), 3), dtype=np.uint8) * 255
    try:
        while True:
            value = sensor.getLidarData()
            detect = con.compute(value)
            con.makeMAT(img, value)
            for rect in detect:
                cv2.rectangle(img,
                (int(rect[0][0]),int(rect[0][2])),
                (int(rect[0][1]),int(rect[0][3])),
                rect[2],
                5,
                cv2.LINE_AA
                )
                cv2.line(img, (int(rect[1][0]),int(rect[1][1])), (int(rect[1][2]), int(rect[1][3])), (127, 127, 127), 1, cv2.LINE_AA)
            cv2.imshow("grid",img)
            cv2.waitKey(1)
            con.init()
            img = np.ones((real_y, real_x+int(real_y*math.sin(30/180*math.pi)/math.cos(30/180*math.pi)), 3), dtype=np.uint8) * 255
            curtime = datetime.datetime.now()
            print("ms :"+str((curtime-oldtime).microseconds/1000))
            oldtime = curtime
    except KeyboardInterrupt:
        exit()
