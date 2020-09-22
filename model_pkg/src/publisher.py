#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

import dynamic_model

def talker(): 
    pub = rospy.Publisher('chatter', String, queue_size=10) # creating a pub with topic -> chatter
    rospy.init_node('talker', anonymous=True) 
    rate = rospy.Rate(5) #  defining rate of publishing ... 10hz
    while not rospy.is_shutdown():
        
	flag, output, measurement = dynamic_model.do_inference_ll() # Inference using lower leg
	result = [str(x.item()) for x in output]
	result = ','.join(result)
        rospy.loginfo('Measured segment: '+ flag)
        rospy.loginfo(result)
 	pub.publish('Measured segment: '+ flag)
 	pub.publish(result)    
	pub.publish("Error : {} cm".format(output[-1].item() - measurement/10))    
	
	flag, output, measurement = dynamic_model.do_inference_ul() # Inference using upper leg
	result = [str(x.item()) for x in output]
	result = ','.join(result)
        rospy.loginfo('Measured segment: '+ flag)
        rospy.loginfo(result)
 	pub.publish('Measured segment: '+ flag)
 	pub.publish(result)    
	pub.publish("Error : {} cm".format(output[-1].item() - measurement/10))  

	flag, output, measurement = dynamic_model.do_inference_la() # Inference using lower arm
	result = [str(x.item()) for x in output]
	result = ','.join(result)
        rospy.loginfo('Measured segment: '+ flag)
        rospy.loginfo(result)
 	pub.publish('Measured segment: '+ flag)
 	pub.publish(result)    
	pub.publish("Error : {} cm".format(output[-1].item() - measurement/10))  

	flag, output, measurement = dynamic_model.do_inference_ua() # Inference using upper arm
	result = [str(x.item()) for x in output]
	result = ','.join(result)
        rospy.loginfo('Measured segment: '+ flag)
        rospy.loginfo(result)
 	pub.publish('Measured segment: '+ flag)
 	pub.publish(result)    
	pub.publish("Error : {} cm".format(output[-1].item() - measurement/10))  
	


        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
