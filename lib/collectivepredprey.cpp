/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * predprey.cpp, include an implementation of the ErPredprey environment

 * This file is part of the python module ErPredprey.so that include the following files:
 * predprey.cpp, predprey.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErPredprey.pxd, ErPredprey.pyx and setupErPredprey.py
 * And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErPredprey.py build_ext –inplace; cp ErPredprey*.so ../bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <locale.h>
#include "predprey.h"
#include "utilities.h"
#include "robot-env.h"

#define SCALE 1.0

#define NUM_PREDATORS 10
#define NUM_PREYS 1
#define FOV PI2 * 3.0 / 2.0
#define NUM_SECTORS 8
#define MAX_DIST 100.0

// the current step
int cstep;
int steps = 1000;
// Number of predators and preys
int npredators = NUM_PREDATORS;
int npreys = NUM_PREYS;
// Field of view of the agent
double fov = FOV;
// Random velocities for robots
bool randVel = false;
// Additional inputs
bool addIn = false;
// Number of internal units
int nhiddens = 10;
// read configuration parameters from file
void readEvoConfig();

// Pointer to the observations
float* cobservation;
// Pointer to the actions
float* caction;
// Pointer to termination flag
int* cdone;
// Pointer to world objects to be rendered
double* dobjects;

void updatePrey(struct robot *cro);
int initCameraSensor(struct robot *cro);
void updateCameraSensor(struct robot *cro);
/*
 * env constructor
 */
Problem::Problem()
{

    struct robot *ro;
    int r;

    // set USA local conventions
    setlocale( LC_ALL, "en-US" );
    // read steps from .ini file
    readEvoConfig();
    // Consistency check between parameters in configuration file
    if (nrobots != (npredators + npreys))
    {
    	printf("FATAL ERROR: num_robots %d != num_predators %d + num_preys %d!!!\n", nrobots, npredators, npreys);
    	exit(-1);
    }
	// create the environment
	initEnvironment();
    // creates the list of robot structures that contain the robot data
    // the first is the predator and the second is the prey
    rob = (struct robot *) malloc(nrobots * sizeof(struct robot));
    //
    // the initRobot function is defined in the robot-env.cpp file
	for (r=0, ro=rob; r < nrobots; r++, ro++)
    {
		
	   initRobot(ro, r, MarXBot);             // initilize robot variable (e.g. the size of the robots' radius, ext).
        // set predators and prey properties
        if (r < npredators)
         {
            ro->color = 1;  // predators are red
            ro->maxSpeed = 500.0 * 0.80;  // predator is 20% slower
         }
        else
        {
            ro->color = 3;  // prey are green
            ro->maxSpeed = 500.0;
        }
	
	   ninputs = 0;
       ninputs += initCameraSensor(ro);     // infrared sensor (8 units)

	   initRobotSensors(ro, ninputs);        // allocate and initialize the robot->sensor vector that contain net->ninputs values and is passed to the function that update the state of the robots' network
		
       ro->motorwheels = 2;                   // define the motors used and set the number of motor neurons
	   ro->motorwheelstype = 0;               // direct controls of the wheels
       noutputs = ro->motorwheels;
		
       ro->motorwheelsid = ninputs + nhiddens;
	}
	
	rng = new RandomGenerator(time(NULL));
	
}


Problem::~Problem()
{
}


/*
 * set the seed
 */
void Problem::seed(int s)
{
    	rng->setSeed(s);
}

void Problem::init()
{
	int r;
	if (randVel)
	{
		// Randomize predator velocity
		for (r=0, ro=rob; r < npredators; r++, ro++)
    		{
    			// Velocity is randomized in the range [maxPredVel / 2, maxPredVel]
    			ro->maxSpeed = rng->getDouble(250.0 * 0.80, 500.0 * 0.80);
    		}
	}
	if (addIn)
	{
		// Do some stuff
	}
}

/*
 * reset the initial condition randomly
 * when seed is different from 0, reset the seed
 */
void Problem::reset()
{
    	int attempts;
	double dx, dy;
	struct robot *ro1;
	struct robot *ro2;
	int r1, r2;
	double cdist, mindist;
	double distfromcenter = 50.0;
	double centerx, centery;

    	cstep = 0;
    	
    	// Center coordinates
    	centerx = worldx / 2.0;
    	centery = worldy / 2.0;

	switch (robottype)
	{
	   	case (Khepera):
	    		distfromcenter = 150.0;
			break;
	   	case (ePuck):
	    		distfromborder = 400.0;
			distfromcenter = 200.0;
			break;
	   	case (MarXBot):
	   	default:
	    		distfromborder = 1000.0;
			distfromcenter = 500.0;
			break;
	}

      	// initial positions and orientations of the robots
	for (r1=0, ro1=rob; r1 < nrobots; r1++, ro1++)
	{
        	mindist = 0.0; attempts = 0;
		while(mindist < (ro1->radius*2+5) && attempts < 100)
		{
          		ro1->dir = rng->getDouble(0.0, PI2);
          		ro1->x = rng->getDouble(centerx - distcenter, centerx + distcenter);
          		ro1->y = fcy + rng->getDouble(centery - distcenter, centery + distcenter);
		  	mindist = 99999;
	      		for (r2=0, ro2=rob; r2 < r1; r2++, ro2++)
	       	{
			  	dx = (ro1->x - ro2->x);
			  	dy = (ro1->y - ro2->y);
			  	cdist = sqrt(dx*dx+dy*dy);
			  	if (cdist < mindist)
				  	mindist = cdist;
		   	}
		 	attempts++;
		}
	}

	// Get observations
	getObs();
	
}


void Problem::copyObs(float* observation)
{
	cobservation = observation;
}

void Problem::copyAct(float* action)
{
	caction = action;
}

void Problem::copyDone(int* done)
{
	cdone = done;
}

void Problem::copyDobj(double* objs)
{
	dobjects = objs;
}

/*
 * update observation vector
 * that contains the predator and prey observation state
 */
void Problem::getObs()
{
    
    struct robot *ro;
    int r;
    int u;
    int s;
    
    u = 0;
    for (r=0, ro=rob; r < nrobots; r++, ro++)
    {
        ro->csensors = ro->sensors;
        updateCameraSensor(ro);
        for(s=0, ro->csensors = ro->sensors; s < ninputs; s++, u++, ro->csensors++)
            cobservation[u] = *ro->csensors;
    }
    
}

/*
 * perform the action, update the state of the environment, update observations, return the predator's reward
 */
double Problem::step()
{

    double dx, dy;
    int x, y;
    double dist;
    struct robot *ro;
    struct robot *oro;
    int r;
    double reward;
    float *cacti;
    double angle1, angle2;
	
    cstep++;
    *cdone = 0;
    reward = 0.0;
    cacti = caction;
    // Pointer to the prey
    oro = rob+npredators;
    for (r=0, ro=rob; r < npredators; r++, ro++)
    {
	  if (updateRobot(ro, cacti) > 0)
          {
        	dx = ro->x - oro->x;
        	dy = ro->y - oro->y;
        	dist = sqrt(dx*dx+dy*dy);
        	if (dist <= 0.000001)
        	{
	  	 	*cdone = 1;
         		reward = 1.0 - ((double) cstep / (double) steps);
        	}
        	
         }
        
      	 if (r < (npredators - 1))
         	cacti = (cacti + noutputs);
    }
    
    // Update prey
    updatePrey(oro);
    
    getObs();
    
    return(reward);

}

int Problem::isDone()
{
	return *cdone;
}

void Problem::close()
{
    	//printf("close() not implemented\n");
}

/*
 * initialize the environment
 */
void
Problem::initEnvironment()

{
    
    int cobj=0;
    
    nenvobjs = 4;    // total number of objects
    worldx = 3000;
    worldy = 3000;
    int f;
    
    envobjs = (struct envobjects *) malloc(nenvobjs * sizeof(struct envobjects));
    
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = 0.0;
    cobj++;
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = 0.0;
    envobjs[cobj].y2 = worldy;
    cobj++;
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = worldx;
    envobjs[cobj].y = 0.0;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = worldy;
    cobj++;
    envobjs[cobj].type = WALL;
    envobjs[cobj].x = 0.0;
    envobjs[cobj].y = worldy;
    envobjs[cobj].x2 = worldx;
    envobjs[cobj].y2 = worldy;
    cobj++;
    
    if (cobj > nenvobjs)
    {
        printf("ERROR: you should allocate more space for environmental objects");
        fflush(stdout);
    }
    
}


/*
 * create the list of robots and environmental objects to be rendered graphically
 */
void Problem::render()
{
    int i;
    int c;
    struct robot *ro;
    double scale = 0.12;
    
    c=0;
    // robots
    for (i=0, ro = rob; i < nrobots; i++, ro++)
    {
        dobjects[c] = 1.0;
        dobjects[c+1] = ro->x * scale;
        dobjects[c+2] = ro->y * scale;
        dobjects[c+3] = ro->radius * scale;
        dobjects[c+4] = 0.0;
        if (i == 0)
          {
            dobjects[c+5] = 1.0; //ro->rgbcolor[0];
            dobjects[c+6] = 0.0; //ro->rgbcolor[1];
            dobjects[c+7] = 0.0; // ro->rgbcolor[2];
		  }
		 else
          {
            dobjects[c+5] = 0.0; //ro->rgbcolor[0];
            dobjects[c+6] = 1.0; //ro->rgbcolor[1];
            dobjects[c+7] = 0.0; // ro->rgbcolor[2];
		  }
        dobjects[c+8] = (ro->x + xvect(ro->dir, ro->radius)) * scale;
        dobjects[c+9] = (ro->y + yvect(ro->dir, ro->radius)) * scale;
        c += 10;
    }
    for(i=0; i < nenvobjs; i++)
    {
        switch(envobjs[i].type)
        {
            case SAMPLEDSCYLINDER:
                dobjects[c] = 3.0;
                dobjects[c+3] = envobjs[i].r;
                dobjects[c+4] = 0.0;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
            case WALL:
                dobjects[c] = 2.0;
                dobjects[c+3] = envobjs[i].x2 * scale;
                dobjects[c+4] = envobjs[i].y2 * scale;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
        }
        dobjects[c+1] = envobjs[i].x * scale;
        dobjects[c+2] = envobjs[i].y * scale;
        dobjects[c+5] = envobjs[i].color[0];
        dobjects[c+6] = envobjs[i].color[1];
        dobjects[c+7] = envobjs[i].color[2];
        c += 10;
    }
    dobjects[c] = 0.0;
}

double Problem::renderScale()
{
	return SCALE;
}

/*
 * reads parameters from the configuration file
 */
void readEvoConfig()
{
    char *s;
    char buff[1024];
    char name[1024];
    char value[1024];
    char *ptr;
    int section;  // 0=before the section 1=in the section 2= after the section
    int param;
    
    section = 0;
    
    FILE* fp = fopen("ErPredprey.ini", "r");
    if (fp != NULL)
    {
        // Read lines
        while (fgets(buff, 1024, fp) != NULL)
        {
            
            //Skip blank lines and comments
            if (buff[0] == '\n' || buff[0] == '#' || buff[0] == '/')
                continue;
            
            //Parse name/value pair from line
            s = strtok(buff, " = ");
            if (s == NULL)
                continue;
            else
                copyandclear(s, name);
            
            s = strtok(NULL, " = ");
            if (s == NULL)
                continue;
            else
                copyandclear(s, value);
            
            // Copy into correct entry in parameters struct
            if (strcmp(name, "maxsteps")==0)
                steps = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "nrobots")==0)
                nrobots = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "npredators")==0)
                npredators = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "npreys")==0)
                npreys = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "fov")==0)
            	fov = strtod(value, &ptr);
            else if (strcmp(name, "randVel")==0)
            {
            	param = (int)strtol(value, &ptr, 10);
            	randVel = (bool)param;
            }
            else if (strcmp(name, "addIn")==0)
            {
            	param = (int)strtol(value, &ptr, 10);
            	addIn = (bool)param;
            }
            else if (strcmp(name, "nhiddens")==0)
            	nhiddens = (int)strtol(value, &ptr, 10);
            //else printf("WARNING: Unknown parameter %s \n", name);
        }
        fclose(fp);
    }
    else
    {
        printf("ERROR: unable to open file ErForaging.ini\n");
        fflush(stdout);
    }
}

void updatePrey(struct robot *cro)
{
	// Check detection of some predators
	// Prey has a field of view (FOV) of 360°
	// Pointer to prey action
	float* cacti;
	// Pointer to robots (i.e., predators)
	struct robot *oro;
	// Sector
	int sector;
	const double sectorSize = PI2 / NUM_SECTORS;
	int csector;
	int s;
	// Distance
	double dist;
	double maxDist; // Maximum distance to decide the direction of motion
	// Intervals
	double minAng, maxAng;
	// Coordinates
	double px, py, ox, oy;
	// Angle
	double angle;
	// Robot idx
	int r;
	// Robot orientation
	double orientation;
	// Random number
	double coin;
	// Flag
	bool found;
	// Copy pointer to action
	cacti = (caction+npredators);
	// With 90% chance, the prey moves forward
	coin = rng->getDouble(0.0, 1.0);
	if (coin > 0.1)
	{
		// Prey moves forward
		cacti[0] = cacti[1] = 1.0;
	}
	else
	{
		// Prey turns
		// Get prey position
		px = cro->x;
		py = cro->y;
		// Get prey orientation
		orientation = cro->dir;
		// Look for the best sector
		maxDist = 0.0;
		sector = -1;
		minAng = orientation - M_PI;
		maxAng = minAng + sectorSize;
		for (r=0, oro=rob; r < npredators; r++, oro++)
		{
			// Get coordinates of the current predator
			ox = oro->x;
			oy = oro->y;
			// Compute distance
			dist = sqrt((px-ox)*(px-ox)+(py-oy)*(py-oy));
			// Check if current predator is sufficiently close to the prey
			if (dist < MAX_DIST)
			{
				// Compute relative angle
				angle = atan2(oy-py, ox-px) - orientation;
				// Set angle in range [-M_PI,M_PI]
				if (angle < -M_PI)
					angle += PI2;
				if (angle > M_PI)
					angle -= PI2;
				// Look for the sector
				csector = -1;
                		minAng = orientation - M_PI;
                		maxAng = minAng + sectorSize;
				s = 0;
				found = false;
				// Loop until sector has not been found
				while ((s < NUM_SECTORS) && !found)
				{
					if ((angle >= minAng) && (angle < maxAng))
					{
						csector = s;
						found = true;
					}
					// Update ranges
					minAng = maxAng;
					maxAng = minAng + sectorSize;
					s++;
				}
				// Check whether the sector has been actually found!!!
				if (!found)
				{
					// Code should never enter here!!!
					printf("FATAL ERROR!!! The %d-th predator has distance %.3f from prey (relative angle = %.3f), but the sector has not been found (sector = %d)\n", r, dist, angle, csector);
					exit(-1);
				}
			}
			// Check whether the distance is higher
			if (dist > maxDist)
			{
				// Update maximum distance
				maxDist = dist;
				// Store sector
				sector = csector;
			}
		}
		cro->dir += 0.0;
		// Prey moves forward
                cacti[0] = cacti[1] = 1.0;
	}
}

/*
 *    initialize the camera sensor
 */
int initCameraSensor(struct robot *cro)
{
    if (cro->idn == 0) printf("Camera[%d]: single color, 8 sectors, distance and orientation \n", nrobots * 2);
    return(nrobots * 2);
}


/*
 *    update the omidirectional camera sensor
 *  assume that the environment include a single coloured object constituted by the other robot
 *  encode angular centre of the color blob produced by the other robot within 8 sectors and the average fraction of pixel stimulated
 */
void updateCameraSensor(struct robot *cro)

{
    
    double dx, dy, dist, angle;
    double angleb;
    int n,r,s;
    struct robot *ro;
    double act[8];
    double adist;
    double sectoracenter;
    double pi8;
    double pi16;
    
    pi8 = PI2 / 8.0;
    pi16 = PI2 / 16.0;
    for (n=0; n < 8; n++)
        act[n] = 0.0;
    for (r=0, ro=rob; r < nrobots; r++, ro++)
    {
        if (cro->idn != ro->idn)
        {
            dx = (cro->x - ro->x);
            dy = (cro->y - ro->y);
            dist = sqrt(dx*dx+dy*dy);
            angle = mangrelr(angv(ro->x, ro->y, cro->x, cro->y), cro->dir);
            for (s=0, sectoracenter = 0.0 + (PI2 / 16.0); s < 8; s++, sectoracenter += pi8)
            {
                angleb = angle;
                if (s == 0 && angle > M_PI)
                    angleb = angle - PI2;
                if (s == 7 && angle < M_PI)
                    angleb = angle + PI2;
                if (fabs(angleb - sectoracenter) < pi8)
                {
                    act[s] = 1.0 - ((fabs(angleb - sectoracenter) / pi8));
                }
                //printf("s %d ang %.2f secta %.2f fabs %.2f -> %.2f\n", s, angle, sectoracenter, fabs(angleb - sectoracenter), act[s]);
            }
            
        }
    }
    // perceived angular blob (maximum value il 0.93 radiants for two adjacent marxbots)
    adist = ((M_PI / 2.0) - atan(dist/cro->radius)) * 2.0;
    //printf("dist %.2f angular blob %.2f\n", dist, adist);
    
    // normalize and copy on the imput vector
    for(n=0; n < 8; n++)
    {
        *cro->csensors = act[n];
        //printf("%.2f\n", act[n]);
        cro->csensors++;
    }
    *cro->csensors = adist;
    //printf("%.2f\n", adist);
    cro->csensors++;
    //printf("end\n");
}

/*
 *    initialize the ground gradient sensor
 */
int initGroundGradSensor(struct robot *cro)
{
    return(5);
}
/*
 *    update the marXbot ground gradient sensors
 *  assume that the color of the area is white at the centre and progressively darker in the periphery
 *  the last value encode the average color detected by the four sensors located at the frontal-left/right and at the back-left/right
 *  the first four values variation of the with respect to the average color
 */
void updateGroundGradSensor(struct robot *cro)

{
    double act[5];
    double gcolor[4];
    double x,y;
    double dx, dy;
    double average;
    double wx, wy;
    double maxdist;
    int s, n;
    double sensordist;
    
    wx = worldx / 2.0;
    wy = worldy / 2.0;
    maxdist = sqrt((wx*wx)+(wy*wy));
    sensordist = cro->radius * 0.9;
    
    //front left
    x = cro->x + xvect(cro->dir - 0.0872 , sensordist);
    y = cro->y + yvect(cro->dir - 0.0872 , sensordist);
    dx = x - wx;
    dy = y - wy;
    gcolor[0] = (sqrt((dx*dx)+(dy*dy)) / maxdist);
    //front right
    x = cro->x + xvect(cro->dir + 0.0872 , sensordist);
    y = cro->y + yvect(cro->dir + 0.0872 , sensordist);
    dx = x - wx;
    dy = y - wy;
    gcolor[1] = (sqrt((dx*dx)+(dy*dy)) / maxdist);
    //rear left
    x = cro->x + xvect(cro->dir - 0.0872 + M_PI, sensordist);
    y = cro->y + yvect(cro->dir - 0.0872 + M_PI, sensordist);
    dx = x - wx;
    dy = y - wy;
    gcolor[2] = (sqrt((dx*dx)+(dy*dy)) / maxdist);
    //rear right
    x = cro->x + xvect(cro->dir + 0.0872 + M_PI, sensordist);
    y = cro->y + yvect(cro->dir + 0.0872 + M_PI, sensordist);
    dx = x - wx;
    dy = y - wy;
    gcolor[3] = (sqrt((dx*dx)+(dy*dy)) / maxdist);
    
    // average value
    act[4] = average = (gcolor[0] + gcolor[1] + gcolor[2] + gcolor[3]) / 4.0;
    // variations
    for(s=0; s < 4; s++)
        act[s] = 0.5 + (gcolor[s] - average) * 10.0;
    
    // normalize and copy on the imput vector
    for(n=0; n < 5; n++)
    {
        *cro->csensors = act[n];
        cro->csensors++;
    }
    
    
}


