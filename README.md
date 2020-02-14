---
description: 운전자 졸음방지 및 위급상황 대처 시스템
---

# FA-REC
## 시스템 구현
1. 얼굴인식을 통해 등록된 얼굴이면 졸음운전 시스템이 실행되고 등록되지 않은 얼굴이면 찍힌 사진을 차주에게 보낸다.  
2. 눈감김이 2초 이상 인식 될 경우 차량 내장 스피커를 통해  위험 알림음을 울린다. 
3. 운전 중 5회 이상 알람이 울릴 시에는 시동이 꺼지기 전까지 계속 알람을 울리게 하여 졸음을 방지한다. 
4. 위험 알림음이 울림에도 운전자가 눈을 뜨는 것이 인식되지 않으면 위급상황이라고 판단한다.
5. 대처 1단계: GPS를 사용해 운전자의 현재 위치와 운전자의  이름 데이터를 기기에 등록되어 있는 측근자에게 메시지를 보낸다.  
   대처 2단계: 비상등을 키고 차량의 속도를 차가 멈출 때까지 조금씩 줄인다.

![Alt text](https://github.com/hiimin/FA-REC/blob/master/fa-rec.png?raw=true)
