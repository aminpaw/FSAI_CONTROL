
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <kdl/frames.hpp>
#include <std_msgs/Int8.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include <kdl/frames.hpp>
#include "std_msgs/Float32.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/Imu.h"
#include <sstream>
#include <math.h>
#include <asurt_msgs/nodeStatus.h>

/*
- Clean parameters
- Add acceleration
- Add velocity callback + test
*/
using std::string;

int dump_var =0;    //for converting int to enums

class car
{
    private:
        
    public:
        double  x_pos,y_pos,theta,
                v_x, v_y, r, 
                s, delta, d,
                v_s, v_delta, v_d,
                a_s, a_delta, a_d;
        
        double  v_dyn_min,v_kin_max;

        double  x_pos_ub, y_pos_ub,theta_ub,
                v_x_ub, v_y_ub, r_ub, 
                s_ub, delta_ub, d_ub,
                v_s_ub, v_delta_ub, v_d_ub,
                a_s_ub, a_delta_ub, a_d_ub;

        double  x_pos_lb,y_pos_lb,theta_lb,
                v_x_lb, v_y_lb, r_lb, 
                s_lb, delta_lb, d_lb,
                v_s_lb, v_delta_lb, v_d_lb,
                a_s_lb, a_delta_lb, a_d_lb;

        double  l, l_f, l_r, 
                i_z, car_mass,
                wheel_r;

        double old_yt_2;

        double current_res_velocity; 


      
};





class PurePursuit_controller : car
{
    public:

        //! Constructor
        PurePursuit_controller();

        //! Compute velocit commands each time new odometry data is received.
        void odometry_cb(nav_msgs::Odometry odom);
        void compute();

        //! Receive path to follow.
        void receivePath_cb(nav_msgs::Path path);
        //! Compute transform that transforms a pose into the robot frame (base_link)
        KDL::Frame transformToBaseLink(const geometry_msgs::Pose& pose,
                                        const geometry_msgs::Transform& tf);
        
       //! Helper founction for computing eucledian distances in the x-y plane.
        template<typename T1>
        double distance(T1 pt1)
        {
            return sqrt(pow(pt1.x,2) + pow(pt1.y,2));
        }

    private:
        double ld_1,ld_2,ld_3,ld_4;
        double ld_1_act,ld_2_act,ld_3_act,ld_4_act;
        double pos_tol;
        double v_des, v_des_ub, v_des_lb ;
        double t_max;
        double traction_b;
        double ld_curvature_min , ld_curvature_max;
        float ld_curvature;
        float curvature_x1,curvature_y1,curvature_x2,curvature_y2,curvature_x3,curvature_y3,    // to be optimized 
                curvature_k1 ,curvature_k2 ,curvature_b ,curvature_a ;
        float tire_constraint_red_y ,tire_constraint_aa_y, tire_constraint_bb_y ,tire_constraint_cc_y;
        float tire_constraint_red_x ,tire_constraint_aa_x, tire_constraint_bb_x ,tire_constraint_cc_x;

        unsigned idx_1,idx_2,idx_3,idx_4;
        nav_msgs::Path path_raw;
        //nav_msgs::Path path;
        bool goal_reached;
        //geometry_msgs::Twist cmd_vel;
        ackermann_msgs::AckermannDriveStamped cmd_acker;
       
        ros::NodeHandle nh,nh_private;
        ros::Subscriber sub_odom, sub_path, sub_path_map;
        //ros::Publisher pub_vel, pub_acker, pub_control_actions_msg_pp_to_NH;
        ros::Publisher pub_control_actions_msg_pp_to_NH;
        ros::Publisher tf_path_pub;
        ros::Publisher lookahead_1_pub;
        ros::Publisher status_pub;

        ros::Subscriber subcontrol_sub ;
        ros::Subscriber Pure_Pursuit_sub;
        
        tf2_ros::Buffer tf_buffer;
        tf2_ros::TransformListener tf_listener;
        tf2_ros::TransformBroadcaster tf_broadcaster;
        geometry_msgs::TransformStamped lookahead_1;
        geometry_msgs::TransformStamped lookahead_2;
        asurt_msgs::nodeStatus status_ready;
        asurt_msgs::nodeStatus status_running;

        string map_frame_id, robot_frame_id, lookahead_frame_id, acker_frame_id;

};

        PurePursuit_controller::PurePursuit_controller():ld_1(1.0), v_des(v_des), pos_tol(0.1), idx_1(0),
                             goal_reached(true), nh_private("~"), tf_listener(tf_buffer),
                             map_frame_id("map"), robot_frame_id("base_link"),
                             lookahead_frame_id("lookahead")
        
        {
            goal_reached = true;
            idx_1 = 0;
            idx_2 = 0;
            idx_3 = 0;
            idx_4 = 0;
            old_yt_2 = 0;
            ld_1_act = 0;
            ld_2_act = 0;
            ld_3_act = 0;
            ld_4_act = 0;
            current_res_velocity = 0;
            ld_curvature = 0;
            
            // setting genral parameters for enums

            nh_private.getParam("/move_command",dump_var);
                
            nh_private.getParam("/states",dump_var);
            nh_private.getParam("/plot",dump_var);

            nh_private.getParam("/map",dump_var);

            nh_private.getParam("/state_estimation",dump_var);

            nh_private.getParam("/system_diagnostic",dump_var);

            nh_private.getParam("/x_pos",x_pos);

            // Get parameters from the parameter server
            //nh_private.param<double>("wheelbase", l, 1.0);
            nh_private.param<double>("lookahead_distance", ld_1, 1.0);
            //nh_private_.param<double>("linear_velocity", v_, 0.1);
            
            nh_private.param<double>("position_tolerance", pos_tol, 0.1);
                
            nh_private.param<string>("map_frame_id", map_frame_id, "map");
            // Frame attached to midpoint of rear axle (for front-steered vehicles).
            nh_private.param<string>("robot_frame_id", robot_frame_id, "base_link");
            // Lookahead frame moving along the path as the vehicle is moving.
            nh_private.param<string>("lookahead_frame_id", lookahead_frame_id, "lookahead_1");
            // Frame attached to midpoint of front axle (for front-steered vehicles).
            nh_private.param<string>("ackermann_frame_id", acker_frame_id, "rear_axle_midpoint");
          
            
            nh_private.getParam("/tire_constraint_bb_y",tire_constraint_bb_y);
            nh_private.getParam("/tire_constraint_cc_y",tire_constraint_cc_y);
            nh_private.getParam("/tire_constraint_bb_x",tire_constraint_bb_x);
            nh_private.getParam("/tire_constraint_cc_x",tire_constraint_cc_x);
            nh_private.getParam("/v_kin_max",v_kin_max);
            nh_private.getParam("/v_dyn_min",v_dyn_min);
            nh_private.getParam("/l",l);
            nh_private.getParam("/l",l);
            nh_private.getParam("/l_f",l_f);
            nh_private.getParam("/l_r",l_r);
            nh_private.getParam("/i_z",i_z);
            nh_private.getParam("/car_mass",car_mass);
            nh_private.getParam("/ld_1",ld_1);
            nh_private.getParam("/ld_2",ld_2);
            nh_private.getParam("/ld_3",ld_3);
            nh_private.getParam("/ld_4",ld_4);
            nh_private.getParam("/pos_tol",pos_tol);
            nh_private.getParam("/v_des",v_des);
            nh_private.getParam("/v_des_ub",v_des_ub);
            nh_private.getParam("/v_des_lb",v_des_lb);
            nh_private.getParam("/t_max",t_max);
            nh_private.getParam("/wheel_r",wheel_r);
            nh_private.getParam("/traction_b",traction_b);
            nh_private.getParam("/ld_curvature_min",ld_curvature_min);
            nh_private.getParam("/ld_curvature_max",ld_curvature_max);

            nh_private.getParam("/y_pos",y_pos);
            nh_private.getParam("/theta",theta);
            nh_private.getParam("/v_x",v_x);
            nh_private.getParam("/v_y",v_y);
            nh_private.getParam("/r",r);
            nh_private.getParam("/s",s);
            nh_private.getParam("/delta",delta);
            nh_private.getParam("/d",d);
            nh_private.getParam("/v_s",v_s);
            nh_private.getParam("/a_s",a_s);
            nh_private.getParam("/v_d",v_d);
            nh_private.getParam("/a_delta",a_delta);
            nh_private.getParam("/a_d",a_d);

            nh_private.getParam("/x_pos_ub",x_pos_ub);
            nh_private.getParam("/y_pos_ub",y_pos_ub);
            nh_private.getParam("/theta_ub",theta_ub);
            nh_private.getParam("/v_x_ub",v_x_ub);
            nh_private.getParam("/v_y_ub",v_y_ub);
            nh_private.getParam("/r_ub",r_ub);
            nh_private.getParam("/s_ub",s_ub);
            nh_private.getParam("/delta_ub",delta_ub);
            nh_private.getParam("/d_ub",d_ub);
            nh_private.getParam("/v_s_ub",v_s_ub);
            nh_private.getParam("/a_s_ub",a_s_ub);
            nh_private.getParam("/v_d_ub",v_d_ub);
            nh_private.getParam("/a_delta_ub",a_delta_ub);
            nh_private.getParam("/a_d_ub",a_d_ub);

            nh_private.getParam("/x_pos_lb",x_pos_lb);
            nh_private.getParam("/y_pos_lb",y_pos_lb);
            nh_private.getParam("/theta_lb",theta_lb);
            nh_private.getParam("/v_x_lb",v_x_lb);
            nh_private.getParam("/v_y_lb",v_y_lb);
            nh_private.getParam("/r_lb",r_lb);
            nh_private.getParam("/s_lb",s_lb);
            nh_private.getParam("/delta_lb",delta_lb);
            nh_private.getParam("/d_lb",d_lb);
            nh_private.getParam("/v_s_lb",v_s_lb);
            nh_private.getParam("/a_s_lb",a_s_lb);
            nh_private.getParam("/v_d_lb",v_d_lb);
            nh_private.getParam("/a_delta_lb",a_delta_lb);
            nh_private.getParam("/a_d_lb",a_d_lb);

             // Populate messages with static data
            lookahead_1.header.frame_id = robot_frame_id;
            lookahead_1.child_frame_id = lookahead_frame_id;

            cmd_acker.header.frame_id = "map";
            ///cmd_acker.drive.jerk = jerk;
            sub_path = nh.subscribe("/planning/path", 1000, &PurePursuit_controller::receivePath_cb, this);
            sub_path_map = nh.subscribe("/planning/map", 1000, &PurePursuit_controller::receivePath_cb, this);
            //sub_path = nh.subscribe("/planner/path", 1, &PurePursuit_controller::receivePath_cb, this);
            sub_odom = nh.subscribe("odometry", 1000, &PurePursuit_controller::odometry_cb, this);
            pub_control_actions_msg_pp_to_NH = nh.advertise<ackermann_msgs::AckermannDriveStamped>("sub_control_actions", 1000);
            tf_path_pub = nh.advertise<nav_msgs::Path>("/control/tf_path", 1000);
            
            lookahead_1_pub = nh.advertise<geometry_msgs::PointStamped>("/control/lookahead_1", 1000);
            status_pub = nh.advertise<asurt_msgs::nodeStatus>("/status/control", 1000);
            status_ready.status = 1;
            status_running.status = 2;
            status_pub.publish(status_ready);
        }

        void PurePursuit_controller::compute()
        {   
            // The velocity commands are computed, each time a new Odometry message is received.
            // Odometry is not used directly, but through the tf tree.

            // Get the current robot pose

                geometry_msgs::TransformStamped tf;

                

                try
                {
                    tf = tf_buffer.lookupTransform(path_raw.header.frame_id, "rear_link", ros::Time(0));
                    //path.poses = geometry_msgs::PoseStamped[path_raw.poses.size()];
                    nav_msgs::Path path;
                    for(int i = 0; i < path_raw.poses.size(); i++){
                        KDL::Frame F_bl_ld1 = transformToBaseLink(path_raw.poses[i].pose, tf.transform);
                        geometry_msgs::PoseStamped pose;
                        pose.pose.position.x = F_bl_ld1.p.x();
                        pose.pose.position.y = F_bl_ld1.p.y();
                        pose.pose.position.z = F_bl_ld1.p.z();
                        F_bl_ld1.M.GetQuaternion(pose.pose.orientation.x,
                                                    pose.pose.orientation.y,
                                                    pose.pose.orientation.z,
                                                    pose.pose.orientation.w);
                        path.poses.push_back(pose);
                    }
                    path.header.frame_id = "rear_link";
                    tf_path_pub.publish(path);

                    for (; idx_1 < path.poses.size(); idx_1++)
                    {
                        if (distance(path.poses[idx_1].pose.position) > ld_1)
                        {   ld_1_act = distance(path.poses[idx_1].pose.position);
                            //ROS_WARN("ld_1_act :[%f]",ld_1_act);
                            // Transformed lookahead to base_link frame is lateral error
                            lookahead_1.transform.translation.x = path.poses[idx_1].pose.position.x;
                            lookahead_1.transform.translation.y = path.poses[idx_1].pose.position.y;
                            lookahead_1.transform.translation.z = path.poses[idx_1].pose.position.z;

                            lookahead_1.transform.rotation.x = path.poses[idx_1].pose.orientation.x;
                            lookahead_1.transform.rotation.y = path.poses[idx_1].pose.orientation.y;
                            lookahead_1.transform.rotation.z = path.poses[idx_1].pose.orientation.z;
                            lookahead_1.transform.rotation.w = path.poses[idx_1].pose.orientation.w;
                        

                            // TODO: See how the above conversion can be done more elegantly
                            // using tf2_kdl and tf2_geometry_msgs
                            idx_2 = idx_1;
                            break;
                        }
                    }                   
                    for (; idx_2 < path.poses.size(); idx_2++)
                    {
                        if (distance(path.poses[idx_2].pose.position) > ld_2)
                        {   ld_2_act = distance(path.poses[idx_2].pose.position);
                            //ROS_WARN("ld_1_act :[%f]",ld_1_act);
                            // Transformed lookahead to base_link frame is lateral error
                            lookahead_2.transform.translation.x = path.poses[idx_2].pose.position.x;
                            lookahead_2.transform.translation.y = path.poses[idx_2].pose.position.y;
                            lookahead_2.transform.translation.z = path.poses[idx_2].pose.position.z;

                            lookahead_2.transform.rotation.x = path.poses[idx_2].pose.orientation.x;
                            lookahead_2.transform.rotation.y = path.poses[idx_2].pose.orientation.y;
                            lookahead_2.transform.rotation.z = path.poses[idx_2].pose.orientation.z;
                            lookahead_2.transform.rotation.w = path.poses[idx_2].pose.orientation.w;
                        

                            // TODO: See how the above conversion can be done more elegantly
                            // using tf2_kdl and tf2_geometry_msgs
                            break;
                        }
                    }
                    

                    if (!path.poses.empty() && idx_1 >= path.poses.size())
                    {
                        // We are approaching the goal,
                        // which is closer than ld
                        // This is the pose of the goal w.r.t. the base_link frame

                        if (fabs(path.poses.back().pose.position.x) <= pos_tol)
                        {
                            // We have reached the goal
                            goal_reached = true;
                            //ROS_WARN("goal reached");
                            
                            /*//         set velocity  = zero         */
                            //cmd_acker.drive.steering_angle = 0.0;
                            
                            cmd_acker.drive.speed=0.0;
                            cmd_acker.drive.steering_angle=0.0;
                           
                            //ROS_WARN("cmd_acker.drive.steering_angle:[%f]", cmd_acker.drive.steering_angle);
                            // Set linear velocity for tracking.
                            //cmd_vel.linear.x = 0; // v_ < v_x 
                            //cmd_acker.drive.speed = 0;    // v_ < v_x 
                            //cmd_acker.header.stamp = ros::Time::now();
                            //v_des = 0.0;
                            /*////////////////////////////////////////*/

                            // Reset the path
                            path = nav_msgs::Path();
                        }
                        else
                        {  // To check
                            //ROS_WARN("goal extend to reached");
                            
                            // We need to extend the lookahead distance
                            // beyond the goal point.
                        
                            // Find the intersection between the circle of radius ld
                            // centered at the robot (origin)
                            // and the line defined by the last path pose
                            /*
                            KDL::Frame F_bl_end(KDL::Rotation::Quaternion(path.poses.back().pose.orientation.x,
                                                            path.poses.back().pose.orientation.y,
                                                            path.poses.back().pose.orientation.z,
                                                            path.poses.back().pose.orientation.w),
                                    KDL::Vector(path.poses.back().pose.position.x,
                                                path.poses.back().pose.position.y,
                                                path.poses.back().pose.position.z));
                            double roll, pitch, yaw;
                            F_bl_end.M.GetRPY(roll, pitch, yaw);
                            double k_end = tan(yaw); // Slope of line defined by the last path pose
                            double l_end = F_bl_end.p.y() - k_end * F_bl_end.p.x();
                            // ROS_WARN("l_end last point [%f]",l_end);
                            // ROS_WARN("k_end last point [%f]",k_end);

                            double a = 1 + k_end * k_end;
                            double b = 2 * l_end;
                            double c = l_end * l_end - ld_1 * ld_1;
                            //re check the abs()
                            double D = sqrt(abs(pow(b,2) - 4*a*c));
                            //look at cuurent_res_velocity
                            double x_ld = (-b + copysign(D,current_res_velocity)) / (2*a);  // v_ < v_x 
                            double y_ld = k_end * x_ld + l_end;
                            // ROS_WARN("b last point [%f]",b);
                            // ROS_WARN("D last point [%f]",D);
                            // ROS_WARN("a last point [%f]",a);
                            // ROS_WARN("c last point [%f]",c);

                            
                            lookahead_1.transform.translation.x = x_ld;
                            lookahead_1.transform.translation.y = y_ld;
                            lookahead_1.transform.translation.z = F_bl_end.p.z();
                            F_bl_end.M.GetQuaternion(lookahead_1.transform.rotation.x,
                                                    lookahead_1.transform.rotation.y,
                                                    lookahead_1.transform.rotation.z,
                                                    lookahead_1.transform.rotation.w);
                            geometry_msgs::Point extendded_point;
                            extendded_point.x = x_ld;
                            extendded_point.y = y_ld;
                            extendded_point.z = F_bl_end.p.z();
                            // ROS_WARN("x_ld last point [%f]",x_ld);
                            // ROS_WARN("y_ld last point [%f]",y_ld);
                            
                            ld_1_act = sqrt(pow(x_ld,2) + pow(y_ld,2) );
                           

                            lookahead_2 = lookahead_1;*/

                            lookahead_1.transform.translation.x = path.poses.back().pose.position.x;
                            lookahead_1.transform.translation.y = path.poses.back().pose.position.y;
                            lookahead_1.transform.translation.z = path.poses.back().pose.position.z;

                            lookahead_1.transform.rotation.x = path.poses.back().pose.orientation.x;
                            lookahead_1.transform.rotation.y = path.poses.back().pose.orientation.y;
                            lookahead_1.transform.rotation.z = path.poses.back().pose.orientation.z;
                            lookahead_1.transform.rotation.w = path.poses.back().pose.orientation.w;
                            lookahead_2 = lookahead_1;
                        }
                    }

                    if (!goal_reached)
                    {
                        // Lateral control
                        cmd_acker.drive.steering_angle = std::min(std::max(atan2(2 * lookahead_1.transform.translation.y * l, pow(ld_1_act,2)),delta_lb),delta_ub); 


                        // Longitudinal

                        /*//       adabtive by  curvature calc           //*/   // to be optimized

                        curvature_x1  =  0 ;          //tf.transform.translation.x;
                        curvature_y1  =  0 ;          //tf.transform.translation.y;

                        curvature_x2 = l ;            // tf.transform.translation.x + l;
                        curvature_y2  = 0;            //  tf.transform.translation.y;

                        curvature_x3 =  lookahead_2.transform.translation.x  ;
                        curvature_y3 =  lookahead_2.transform.translation.y  ;
                        
                        curvature_k1 = 0.5*(curvature_x1*curvature_x1 + curvature_y1*curvature_y1 - curvature_x2*curvature_x2 -curvature_y2*curvature_y2)/(curvature_x1-curvature_x2);
                        curvature_k2 = (curvature_y1 - curvature_y2)/(curvature_x1 - curvature_x2);
                        curvature_b = 0.5*(pow(curvature_x2,2) -2*curvature_x2*curvature_k1 + curvature_y2*curvature_y2 - curvature_x3*curvature_x3 + 2*curvature_x3*curvature_k1 - curvature_y3*curvature_y3)/(curvature_x3*curvature_k2 -curvature_y3 +curvature_y2 -curvature_x2*curvature_k2);
                        curvature_a = curvature_k1- curvature_k2*curvature_b;
                        ld_curvature = sqrt(pow((curvature_x1-curvature_a),2)+ pow((curvature_y1-curvature_b),2));


                        if(isnan(ld_curvature)){
                        ld_curvature    =   ld_curvature_min ;        //  maximum velocity in p_p
                        }
                        
                        
                        cmd_acker.drive.speed = std::min(std::max(v_kin_max + ((v_dyn_min - v_kin_max) / (ld_curvature_max - ld_curvature_min)) * (ld_curvature - ld_curvature_min),v_kin_max),v_dyn_min);  //map "v_kin_max : v_dyn_min to r_min : r_max"       
                        
                    }

                    // Publish the lookahead target transform.
                    lookahead_1.header.stamp = ros::Time::now();
                    //tf_broadcaster.sendTransform(lookahead_1);        // uncomment if path was sent with pose and rotation
                    
                    // Publish the velocities
                    //pub_vel.publish(cmd_vel);
                    
                    // Publish ackerman steering setpoints
                    //pub_acker.publish(cmd_acker);

                    // publish control msg to navigation handler 
                    status_pub.publish(status_running);
                    pub_control_actions_msg_pp_to_NH.publish(cmd_acker);
                    
                    
                    // ROS_WARN("ld_1_act :[%f]",ld_1_act);
                    // ROS_WARN("ld_2_act :[%f]",ld_2_act);
                    // ROS_WARN("ld_3_act :[%f]",ld_3_act);
                    // ROS_WARN("ld_4_act :[%f]",ld_4_act);
                    // ROS_WARN("pid.limMax :[%f]", pid.limMax);
                    // ROS_WARN("pid.limMin :[%f]", pid.limMin);
                    // ROS_WARN("pid.out :[%f]", pid.out);
                    // ROS_WARN("ld_curvature :[%f]", ld_curvature);
                    // ROS_WARN("control_actions_msg.v_des :[%f]", control_actions_msg.v_des);
                    // ROS_WARN("control_actions_msg.steering_angel :[%f]",control_actions_msg.steering_angel);
                    // ROS_WARN("===================================================================");
                    geometry_msgs::PointStamped lookahead_1_msg;
                    lookahead_1_msg.header.frame_id = "rear_link";
                    lookahead_1_msg.point.x = lookahead_1.transform.translation.x;
                    lookahead_1_msg.point.y = lookahead_1.transform.translation.y;
                    lookahead_1_pub.publish(lookahead_1_msg);
                }
                catch (tf2::TransformException &ex)
                {
                    //ROS_WARN_STREAM(ex.what());
                    // ROS_WARN("catchhhhhhhhhh");
                }
           

        }

        void PurePursuit_controller::odometry_cb(nav_msgs::Odometry odom)
        {
            // TODO: use ros_can message
            current_res_velocity = sqrt(pow(odom.twist.twist.linear.x ,2)+pow(odom.twist.twist.linear.y, 2));
            
        }       
    
        void PurePursuit_controller::receivePath_cb(nav_msgs::Path new_path)
        {
            // When a new path received, the previous one is simply discarded
            // It is up to the planner/motion manager to make sure that the new
            // path is feasible.
            // Callbacks are non-interruptible, so this will
            // not interfere with velocity computation callback.

            path_raw = new_path;
            idx_1 = 0;
            idx_2 = 0;
            idx_3 = 0;
            idx_4 = 0;
            
            if (new_path.poses.size() > 0)
            {
                goal_reached = false;
            }
            else
            {
                goal_reached = true;
            }        
        }

         KDL::Frame PurePursuit_controller::transformToBaseLink(const geometry_msgs::Pose& pose,const geometry_msgs::Transform& tf)
        {   
            // Pose in global (map) frame
            KDL::Frame F_map_pose(KDL::Rotation::Quaternion(pose.orientation.x,
                                                            pose.orientation.y,
                                                            pose.orientation.z,
                                                            pose.orientation.w),
                                    KDL::Vector(pose.position.x,
                                                pose.position.y,
                                                pose.position.z));

            // Robot (base_link) in global (map) frame
            KDL::Frame F_map_tf(KDL::Rotation::Quaternion(tf.rotation.x,
                                                            tf.rotation.y,
                                                            tf.rotation.z,
                                                            tf.rotation.w),
                                KDL::Vector(tf.translation.x,
                                            tf.translation.y,
                                            tf.translation.z));

            // TODO: See how the above conversions can be done more elegantly
            // using tf2_kdl and tf2_geometry_msgs

            return F_map_tf.Inverse()*F_map_pose;
        }







int main(int argc, char**argv)
{
  ros::init(argc, argv, "pure_pursuit");

  PurePursuit_controller controller;
  ros::Rate rate(30);
  while (ros::ok())
  {
    controller.compute();

    ros::spinOnce();
    rate.sleep();    
  }
  
  return 0;
}