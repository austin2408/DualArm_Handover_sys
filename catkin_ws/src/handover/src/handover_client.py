#!/usr/bin/env python3

import rospy
import smach
import smach_ros
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
import sys

from smach_tutorials.msg import TestAction, TestGoal
from actionlib import *
from actionlib_msgs.msg import *

def static_passive():
    rospy.init_node('handover_client')

    sm0 = smach.StateMachine(outcomes=['succeeded','aborted','preempted', 'End'])

    with sm0:
        smach.StateMachine.add('Init',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=0)),
                               {'succeeded':'Detect','aborted':'Init'})

        smach.StateMachine.add('Detect',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=1)),
                               {'succeeded':'Move_to','aborted':'Detect'})

        smach.StateMachine.add('Move_to',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=2)),
                               {'succeeded':'Grasp_back','aborted':'Move_to'})

        smach.StateMachine.add('Grasp_back',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=3)),
                               {'succeeded':'End','aborted':'Grasp_back'})

    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm0, '/SM_ROOT')
    sis.start()

    outcome = sm0.execute()

    rospy.spin()
    sis.stop()

def dynamic_passive():
    rospy.init_node('handover_client')
    r = TriggerRequest()

    try:
        switch = rospy.ServiceProxy("/handover_server/switch_loop", Trigger)
        resp = switch(r)
    except rospy.ServiceException as exc:
        print("service did not process request: " + str(exc))

    sm0 = smach.StateMachine(outcomes=['succeeded','aborted','preempted', 'End'])

    with sm0:
        smach.StateMachine.add('Init',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=0)),
                               {'succeeded':'Detect','aborted':'Init'})

        smach.StateMachine.add('Detect',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=1)),
                               {'succeeded':'Move_to','aborted':'Detect'})

        smach.StateMachine.add('Move_to',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=2)),
                               {'succeeded':'Check_dis','aborted':'Move_to'})

        smach.StateMachine.add('Check_dis',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=4)),
                               {'succeeded':'Grasp_back','aborted':'Detect'})

        smach.StateMachine.add('Grasp_back',
                               smach_ros.SimpleActionState('handover_action', TestAction,
                               goal = TestGoal(goal=3)),
                               {'succeeded':'End','aborted':'Grasp_back'})

    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm0, '/SM_ROOT')
    sis.start()

    outcome = sm0.execute()

    rospy.spin()
    sis.stop()
    
if __name__ == '__main__':
    # dynamic_passive()
    static_passive()
