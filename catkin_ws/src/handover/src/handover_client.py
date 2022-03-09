#!/usr/bin/env python3

import rospy
import smach
import smach_ros

from smach_tutorials.msg import TestAction, TestGoal
from actionlib import *
from actionlib_msgs.msg import *

def main():
    rospy.init_node('pick_and_place_client')

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
                               goal = TestGoal(goal=4)),
                               {'succeeded':'End','aborted':'Grasp_back'})

    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm0, '/SM_ROOT')
    sis.start()

    outcome = sm0.execute()

    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
