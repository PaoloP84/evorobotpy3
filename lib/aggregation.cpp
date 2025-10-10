/*
 * discrim.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include "robot-env.h"
#include "aggregation.h"
#include "utilities.h"

#define SCALE 1.5

// the current step
int cstep;
int steps = 1000;

// Pointer to the observations
float* cobservation;
// Pointer to the actions
float* caction;
// Pointer to termination flag
int* cdone;
// Pointer to world objects to be rendered
double* dobjects;

int robottype = MarXBot;	// the robot we are using
int nnests;			// number of nest elements
double nestdist;            	// minimum nest distance
double robotnestdist;       	// minimum distance between robot and nest
double robotdist;           	// minimum robot distance
double *robotsdist;		// matrix containing robot distances;
int *robotsbydistance;		// matrix containing the id of the robots ordered by the inverse of the distance
int nhiddens = 10;		// number of internal units
int afunction = 1;		// activation function
double wsize = 5000.0;		// world size
double asize;			// maximum nest size
double msize;			// minimum nest size
double nestBonus = 100.0;	// bonus awarded for the ability to reach a nest
double kdist = 1.0;		// coefficient weighting the distance component of fitness
bool robotOnNest[100];		// flag whether the robot is on the nest
int nrobotsOnNest[100];	// Number of robots on a given nest
double* clstMetric;		// Cluster metric
double* dispMetric;		// Dispersion metric

// Static methods (for sensors and other)
void readEvoConfig();
double calcNestRadius(int nr);
int calcRobots(double radius);
void updateRobotDistances();
int initCameraSensorRFB(struct robot *cro);
void updateCameraAddBlob(double *cb, int *nb, double color, double dist, double brangel, double branger);
void updateCameraSensorRFB(struct robot *cro, int *rbd, int noutputs);
int initGroundSensor(struct robot *cro);
void updateGroundSensor(struct robot *cro);

/*
 * env constructor
 */
Problem::Problem()
{

    	struct robot *ro;
    	int r;
	
	// set USA local conventions
	setlocale( LC_ALL, "en-US" );
	// read parameters from the .ini file
	readEvoConfig();
	
	// create and configure the environment
    	initEnvironment();

   	// create the robots, the networks, and initialize the sensors
    	rob = (struct robot *) malloc(nrobots * sizeof(struct robot));
    	// initialize the robots and the networks
	for (r=0, ro=rob; r < nrobots; r++, ro++)
	{
		// initialize the robots
      		initRobot(ro, r, robottype);
      		// Set robot max speed
      		switch(robottype)
      		{
      			case Khepera:
      				ro->maxSpeed = 144.0;
      				break;
      			case ePuck:
      				ro->maxSpeed = 200.0;
      				break;
      			case MarXBot:
      			default:
      				ro->maxSpeed = 500.0;
      				break;
      		}
      		// Initial color		
      		ro->color = 4;  // frontal and rear side are red and blue, respectively

      		// initialize sensors
      		ninputs = 0;
      		ninputs += initInfraredSensor(ro);
      		ninputs += initCameraSensorRFB(ro);
      		ninputs += initGroundSensor(ro);
      		
      		initRobotSensors(ro, ninputs);        // allocate and initialize the robot->sensor vector that contain net->ninputs values and is passed to the function that update the state of the robots' network
      		// initialize motors
      		ro->motorwheels = 2;
      		ro->motorwheelstype = 1; // encoding of speed and rotation
      		ro->motorleds = 2;

      		// set the id number of motors neurons
      		ro->motorwheelsid = ninputs + nhiddens;
      		ro->motorledsid = ro->motorwheelsid + ro->motorwheels;
      		noutputs = ro->motorwheels + ro->motorleds;
	}
	
	// Update target areas (nests) size
	asize = calcNestRadius(nrobots);
	// Compute the minimum nest size
	msize = (rob->radius * 2.0); // The minimum size to have a robot in a nest
	// Metrics
	clstMetric = (double*)malloc(steps * sizeof(double));
	dispMetric = (double*)malloc(steps * sizeof(double));
	
	robotsdist = (double*)malloc((nrobots * nrobots) * sizeof(double));
	robotsbydistance = (int*)malloc((nrobots * nrobots) * sizeof(int));
	
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

/*
 * reset the initial condition randomly
 * when seed is different from 0, reset the seed
 */
void Problem::reset()
{
	int s;
	double dx, dy;
	struct robot *ro1;
	struct robot *ro2;
	int r1, r2;
	double fcx, fcy, fcr;
	double distfromborder = 100.0;
	bool placed;
	double dist;
	int nest;
	int other;
	double nestPos[100];
	double minsize;
	int nthrobots; // Theoretical number of robots that can stay in a nest
	int nleftrobots; // Remaining robots to be placed

	switch (robottype)
	{
	   	case (Khepera):
	    		distfromborder = asize + 200.0;
			break;
	   	case (ePuck):
	    		distfromborder = asize + 200.0;
			break;
	   	case (MarXBot):
	   	default:
	    		distfromborder = asize + 200.0;
			break;
	}
	
	minsize = msize; // Set minimum size
	nthrobots = 0;
	
	// Nests
	if (nnests == 1)
	{
		fcx = rng->getDouble(distfromborder, worldx - distfromborder);
      		fcy = rng->getDouble(distfromborder, worldy - distfromborder);
      		// Radius is the maximum possible
		fcr = asize;
		// Store values
		envobjs[4].x = nestPos[0] = fcx;
		envobjs[4].y = nestPos[1] = fcy;
		envobjs[4].r = nestPos[2] = fcr;
		nrobotsOnNest[0] = 0;
	}
	else
	{
		for (nest = 0; nest < nnests; nest++)
		{
			placed = false;
			fcx = fcy = fcr = -1.0; // Invalid value
			while (!placed)
			{
				// Extract random position and radius
				fcx = rng->getDouble(distfromborder, worldx - distfromborder);
      				fcy = rng->getDouble(distfromborder, worldy - distfromborder);
      				fcr = rng->getDouble(minsize, asize);
      				placed = true;
				other = 0;
				while ((other < nest) && placed)
				{
				  	// Compute distance with already placed nests
				  	dx = (fcx - nestPos[3 * other]);
				  	dy = (fcy - nestPos[3 * other + 1]);
				  	dist = sqrt(dx*dx+dy*dy);
				  	if (dist < nestdist)
				    		placed = false;
				  	other++;
				}
			}
			envobjs[4 + nest].x = nestPos[3 * nest] = fcx;
			envobjs[4 + nest].y = nestPos[3 * nest] = fcy;
			envobjs[4 + nest].r = nestPos[3 * nest] = fcr;
			nthrobots += calcRobots(fcr);
			// Update minimum size
			if (nest == (nnests - 2))
			{
				// Check the number of robots
				if (nthrobots < nrobots)
				{
					nleftrobots = (nrobots - nthrobots);
					if (nleftrobots > 2)
					{
						minsize = calcNestRadius(nleftrobots);
					}
				}
			}
			nrobotsOnNest[nest] = 0;
		}
	}

      	// initial positions and orientations of the robots
	for (r1=0, ro1=rob; r1 < nrobots; r1++, ro1++)
	{
		placed = false;
		while (!placed)
      		{
      			ro1->dir = rng->getDouble(0.0, PI2);
        		ro1->x = rng->getDouble(200.0, wsize - 200.0);
        		ro1->y = rng->getDouble(200.0, wsize - 200.0);
        		placed = true;
        		// Check whether robot position overlaps with nests
        		nest = 0;
        		while ((nest < nnests) && placed)
        		{
          			// Distance from nest
          			dx = (ro1->x - nestPos[3 * nest]);
          			dy = (ro1->y - nestPos[3 * nest + 1]);
          			dist = sqrt(dx * dx + dy * dy);
          			if (dist < robotnestdist)
            				placed = false;
          			nest++;
        		}
        		if (placed)
        		{
          			// Check whether robot position overlaps with other robots
          			for (r2=0, ro2=rob; r2 < r1; r2++, ro2++)
          			{
            				dx = (ro1->x - ro2->x);
            				dy = (ro1->y - ro2->y);
            				dist = sqrt(dx*dx+dy*dy);
            				if (dist < robotdist)
              					placed = false;
          			}
        		}
     		}
        	ro1->alive = true;
		ro1->energy = 1.0;
		robotOnNest[r1] = false;
		for (s = 0, ro1->csensors = ro1->sensors; s < ninputs; s++, ro1->csensors++)
			*ro1->csensors = 0.0;
		//printf("Robot %d (attempts %d): (%lf,%lf)\n", r1, attempts, ro1->x, ro1->y);
	}

	// Get observations
	getObs();	
	
	cstep = 0;
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
    	int* rbd;
    
    	u = 0;
    
    	for (r=0, ro=rob, rbd=robotsbydistance; r < nrobots; r++, ro++, rbd = (rbd + nrobots))
    	{
    		// update robots distances
    		updateRobotDistances();
        	ro->csensors = ro->sensors;
        	updateInfraredSensor(ro);
        	updateCameraSensorRFB(ro, rbd, noutputs);
        	updateGroundSensor(ro);
        	//printf("cstep %d - robot %d: ", cstep, r);
        	for(s=0, ro->csensors = ro->sensors; s < ninputs; s++, ro->csensors++)
        	{
           		cobservation[u] = *ro->csensors;
           		//printf("%.3f ", cobservation[u]);
           		u++;
           	}
           	//printf("\n");
    	}
}

/*
 * perform the action, update the state of the environment, update observations, return the predator's reward
 */
double Problem::step()
{

    	double dx, dy;
    	double dist;
    	struct robot *ro, *ro2;
    	int r;
    	double reward;
    	float *cacti;
    	int nest;
    	bool onNest;
    	int rr;
    	double robotradius = 1.0;
    	const double maxDist = sqrt(2.0 * wsize * wsize);
    	double cluster;
    	int nclusters;
    	double dispersion;
    	double cogx, cogy;
    	double fit[1000];
	double tfit;
	double cdist;
	double mydist;

    	*cdone = 0;
    	reward = 0.0;
    	for (nest = 0; nest < nnests; nest++)
    		nrobotsOnNest[nest] = 0;
    	for (r=0, ro=rob, cacti=caction; r < nrobots; r++, ro++, cacti = (cacti + noutputs))
    	{
    		onNest = false;
    		if (r == 0)
            		robotradius = ro->radius;
		updateRobot(ro, cacti);
    		for (nest = 0; nest < nnests; nest++)
            	{
              		dx = ro->x - envobjs[4 + nest].x;
              		dy = ro->y - envobjs[4 + nest].y;
              		cdist = sqrt((dx*dx)+(dy*dy));
			if (cdist < envobjs[4 + nest].r)
              		{
                		onNest = true;
                		// Update the number of robots on that nest
                		nrobotsOnNest[nest]++;
              		}
              	}
              	if (onNest)
              		robotOnNest[r] = true;
            	else
              		robotOnNest[r] = false;
    	}
    
	getObs();
	
	// Compute metrics
	// Cluster
	cluster = 0.0;
	nclusters = 0;
	for (nest = 0; nest < nnests; nest++)
        {
        	double nmax;
        	double circ;
        	if (nrobotsOnNest[nest] > 0)
        	{
			circ = PI2 * envobjs[4 + nest].r;
			nmax = circ / (2.0 * robotradius);
			cluster += (nrobotsOnNest[nest] / nmax);
			nclusters++;
		}
	}
	if (nclusters > 0)
		cluster /= nclusters;
	// Update array
	clstMetric[cstep] = cluster;
	// Dispersion
	dispersion = 0.0;
	// Compute COG (Center Of Gravity)
	cogx = 0.0;
	cogy = 0.0;
	for (r=0, ro=rob; r < nrobots; r++, ro++)
	{
		cogx += ro->x;
		cogy += ro->y;
	}
	cogx /= nrobots;
	cogy /= nrobots;
	// Now compute dispersion metric
	for (r=0, ro=rob; r < nrobots; r++, ro++)
	{
		dx = (cogx - ro->x);
		dy = (cogy - ro->y);
		dist = sqrt(dx * dx + dy * dy);
		dispersion += (dist * dist);
	}
	dispersion /= (4.0 * robotradius * robotradius);
	dispMetric[cstep] = dispersion;
	cstep++;
    	// Episode ends if the number of performed steps exceeds the time limit
	if (cstep >= steps)
	{
		// Compute fitness
		tfit = 0.0;
		for (r=0, ro=rob; r < nrobots; r++, ro++)
		{
			fit[r] = 0.0;
			if (robotOnNest[r])
        			fit[r] += nestBonus; // Bonus
			// Compute average distance with swarm-mates
			cdist = 0.0;
          		for (rr = 0, ro2 = rob; rr < nrobots; rr++, ro2++)
          		{
            			// Distance with others
            			dx = (ro->x - ro2->x);
            			dy = (ro->y - ro2->y);
            			mydist = sqrt((dx*dx)+(dy*dy));
            			cdist += (mydist / maxDist);
			}
			cdist /= (nrobots - 1);
			// Add component rewarding for swarm distance minimization
			fit[r] += kdist * (1.0 - cdist);
        		// Update episode fitness
        		tfit += fit[r];
      		}
      		// Normalize episode fitness over the number of robots
      		tfit /= nrobots;
      		// Update fitness
      		reward = tfit;
		*cdone = 1.0;
	}
    
    	return reward;

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
 * create the list of robots and environmental objects to be rendered graphically
 */
void Problem::render()
{
    	int i;
    	int c;
    	struct robot *ro;
    	double scale = 0.1;
    	bool redLed;
    	bool blueLed;
    
    	c=0;
    	// Environmental objects
    	for(i=0; i < nenvobjs; i++)
    	{
        	switch(envobjs[i].type)
        	{
            		case STARGETAREA:
                		dobjects[c] = 3.0;
                		dobjects[c+3] = envobjs[i].r * scale;
                		dobjects[c+4] = 0.0;
                		dobjects[c+8] = 0.0;
                		dobjects[c+9] = 0.0;
                		break;
            		case WALL:
            		default:
                		dobjects[c] = 2.0;
               		dobjects[c+3] = envobjs[i].x2 * scale;
                		dobjects[c+4] = envobjs[i].y2 * scale;
                		dobjects[c+8] = 0.0;
                		dobjects[c+9] = 0.0;
                		break;
        	}
        	dobjects[c+1] = envobjs[i].x * scale;
        	dobjects[c+2] = envobjs[i].y * scale;
        	dobjects[c+5] = envobjs[i].color[0] * 255.0;
        	dobjects[c+6] = envobjs[i].color[1] * 255.0;
        	dobjects[c+7] = envobjs[i].color[2] * 255.0;
        	c += 10;
    	}
    	// robots
    	for (i=0, ro = rob; i < nrobots; i++, ro++)
    	{
    		redLed = false;
    		blueLed = false;
        	dobjects[c] = 1.0;
        	dobjects[c+1] = ro->x * scale;
        	dobjects[c+2] = ro->y * scale;
		dobjects[c+3] = ro->radius * scale;
		dobjects[c+4] = 0.0;
		switch (ro->color)
		{
			case 0:
			default:
				dobjects[c+5] = 0.0; //ro->rgbcolor[0];
		    		dobjects[c+6] = 0.0; //ro->rgbcolor[1];
		    		dobjects[c+7] = 0.0; // ro->rgbcolor[2];
		    		break;
		    	case 1:
            			dobjects[c+5] = 255.0; //ro->rgbcolor[0];
		    		dobjects[c+6] = 0.0; //ro->rgbcolor[1];
		    		dobjects[c+7] = 0.0; // ro->rgbcolor[2];
		    		redLed = true;
		    		break;
		    	case 2:
		    		dobjects[c+5] = 0.0; //ro->rgbcolor[0];
		    		dobjects[c+6] = 0.0; //ro->rgbcolor[1];
		    		dobjects[c+7] = 255.0; // ro->rgbcolor[2];
		    		blueLed = true;
		    		break;
		    	case 3:
            			dobjects[c+5] = 0.0; //ro->rgbcolor[0];
		    		dobjects[c+6] = 255.0; //ro->rgbcolor[1];
		    		dobjects[c+7] = 0.0; // ro->rgbcolor[2];
		    		break;
		    	case 4:
		    		dobjects[c+5] = -1.0; //ro->rgbcolor[0];
		    		dobjects[c+6] = -1.0; //ro->rgbcolor[1];
		    		dobjects[c+7] = -1.0; // ro->rgbcolor[2];
		    		redLed = true;
		    		blueLed = true;
		    		break;
		}
       	dobjects[c+8] = (ro->x + xvect(ro->dir, ro->radius)) * scale;
       	dobjects[c+9] = (ro->y + yvect(ro->dir, ro->radius)) * scale;
        	c += 10;
        	// LEDs (if active)
        	if (redLed)
        	{
        		dobjects[c] = 5.0;
        		dobjects[c+1] = ro->x * scale;//(ro->x + xvect(ro->dir, ro->radius / 2.0)) * scale;
        		dobjects[c+2] = ro->y * scale;//(ro->y + yvect(ro->dir, ro->radius / 2.0)) * scale;
			dobjects[c+3] = ro->radius * scale;//ro->radius / 5.0 * scale;
			dobjects[c+4] = ro->dir;
			dobjects[c+5] = 255.0; //ro->rgbcolor[0];
		    	dobjects[c+6] = 0.0; //ro->rgbcolor[1];
		    	dobjects[c+7] = 0.0;
		    	dobjects[c+8] = (ro->x + xvect(ro->dir, ro->radius)) * scale;
		    	dobjects[c+9] = (ro->y + yvect(ro->dir, ro->radius)) * scale;
		    	c += 10;
        	}
        	if (blueLed)
        	{
        		dobjects[c] = 6.0;
        		dobjects[c+1] = ro->x * scale;//(ro->x - xvect(ro->dir, ro->radius / 2.0)) * scale;
        		dobjects[c+2] = ro->y * scale;//(ro->y - yvect(ro->dir, ro->radius / 2.0)) * scale;
			dobjects[c+3] = ro->radius * scale;//ro->radius / 5.0 * scale;
			dobjects[c+4] = ro->dir;
			dobjects[c+5] = 0.0; //ro->rgbcolor[0];
		    	dobjects[c+6] = 0.0; //ro->rgbcolor[1];
		    	dobjects[c+7] = 255.0;
		    	dobjects[c+8] = (ro->x + xvect(ro->dir, ro->radius)) * scale;
		    	dobjects[c+9] = (ro->y + yvect(ro->dir, ro->radius)) * scale;
		    	c += 10;
        	}
    	}
    	dobjects[c] = 0.0;
}

double Problem::renderScale()
{
	return SCALE;
}

/*
 * initialize the environment
 */
void Problem::initEnvironment()
{

	int cobj=0;
	int f;
	
    	nenvobjs = 4 + nnests;	// total number of objects
	switch (robottype)
	{
	case (Khepera):
	    	asize = 120.0;
		break;
	case (ePuck):
	    	asize = 160.0;
		break;
	case (MarXBot):
	default:
	    	asize = 400.0;
		break;
	}
	
	worldx = wsize;
	worldy = wsize;
	
	envobjs = (struct envobjects *) malloc(nenvobjs * sizeof(struct envobjects));
	
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = 0.0;
	envobjs[cobj].color[0] = 0.0;
	envobjs[cobj].color[1] = 0.0;
	envobjs[cobj].color[2] = 0.0;
	cobj++;
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = 0.0;
	envobjs[cobj].y2 = worldy;
	envobjs[cobj].color[0] = 0.0;
	envobjs[cobj].color[1] = 0.0;
	envobjs[cobj].color[2] = 0.0;
	cobj++;
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = worldx;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = worldy;
	envobjs[cobj].color[0] = 0.0;
	envobjs[cobj].color[1] = 0.0;
	envobjs[cobj].color[2] = 0.0;
	cobj++;
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = worldy;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = worldy;
	envobjs[cobj].color[0] = 0.0;
	envobjs[cobj].color[1] = 0.0;
	envobjs[cobj].color[2] = 0.0;
	cobj++;
	for(f=0; f < nnests; f++)
	{
	  	envobjs[cobj].type = STARGETAREA;
	  	envobjs[cobj].x = 400.0;
	  	envobjs[cobj].y = 400.0;
	  	envobjs[cobj].r = asize;
	  	envobjs[cobj].color[0] = 0.5;
	  	envobjs[cobj].color[1] = 0.5;
	  	envobjs[cobj].color[2] = 0.5;
	  	cobj++;
	}
	
	if (cobj > nenvobjs)
	{
		printf("ERROR: you should allocate more space for environmental objects");
		fflush(stdout);
	}
	
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
    
    // Setting default values for distances (to avoid any numerical issue)
    nestdist = 1000.0;
    robotnestdist = 500.0;
    robotdist = 200.0;
    
    FILE* fp = fopen("ErAggregation.ini", "r");
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
            else if (strcmp(name, "robottype")==0)
            	robottype = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "wsize")==0)
            	wsize = strtod(value,&ptr);
            else if (strcmp(name, "nnests")==0)
            	nnests = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "nestdist")==0)
            	nestdist = strtod(value, &ptr);
            else if (strcmp(name, "robotnestdist")==0)
            	robotnestdist = strtod(value, &ptr);
            else if (strcmp(name, "robotdist")==0)
            	robotdist = strtod(value, &ptr);
            else if (strcmp(name, "nestbonus")==0)
           	nestBonus = strtod(value, &ptr);
            else if (strcmp(name, "kdist")==0)
            	kdist = strtod(value, &ptr);
            else if (strcmp(name, "nhiddens")==0)
            	nhiddens = (int)strtol(value, &ptr, 10);
            else if (strcmp(name, "afunction")==0)
                afunction = (int)strtol(value, &ptr, 10);
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

double calcNestRadius(int nr)
{
	double r;
	if (nr <= 0)
	{
		printf("Invalid number of robots %d!!\n", nr);
		exit(-1);
	}
	if (nr == 1)
		r = rob->radius * 2.0;
	else
		r = rob->radius * (1.0 + (1.0 / sin((2.0 * M_PI) / (2.0 * nr))));
	return r;
}

int calcRobots(double radius)
{
	int n;
	double num;
	if (radius <= 0.0)
	{
		printf("Invalid radius %lf\n", radius);
		exit(-1);
	}
	// To compute the number of robots that can occupy the area, we must invert the formula computing the radius of thea area given the robots
	// Compute ratio between radii
	double r_ratio = radius / rob->radius;
	// Now compute argument of arcsin operation...
	// Remember that r = rob->radius * (1.0 + (1.0 / sin((2.0 * M_PI) / (2.0 * nr))))
	// Therefore:
	// 1) r / rob->radius = 1.0 + (1.0 / sin((2.0 * M_PI) / (2.0 * nr)))
	// 2) (r / rob->radius - 1.0) = 1.0 / sin((2.0 * M_PI) / (2.0 * nr))
	// 3) 1.0 / (r / rob->radius - 1.0) = sin((2.0 * M_PI) / (2.0 * nr))
	// 4) arcsin((1.0 / (r / rob->radius - 1.0))) = M_PI / nr
	// 5) nr = M_PI / arcsin((1.0 / (r / rob->radius - 1.0)))
	double arg = asin(1.0 / (r_ratio - 1.0));
	num = M_PI / arg;
	n = ((double)num);
	return n;
}

/*
 *  update the power-distances between robots and the matrix of nearest robots
 */
void updateRobotDistances()
{
    struct robot *ro1;
    struct robot *ro2;
    int r1, r2, r3;
    double *cdist;
    double *ccdist;
    double smallerdist;
    int smallerid;
    int *rbd;
    double *rbddist;
    int remaining;


    // update the matrix of distances
    cdist = robotsdist;
    for (r1=0, ro1=rob; r1 < nrobots; r1++, ro1++)
    {
      for (r2=0, ro2=rob; r2 < r1; r2++, ro2++)
      {
        *cdist = ((ro1->x - ro2->x)*(ro1->x - ro2->x)) + ((ro1->y - ro2->y)*(ro1->y - ro2->y));
        *(robotsdist + (r1 + (r2 * nrobots))) = *cdist;
        cdist++;
      }
      *cdist = 9999999999.0; // distance from itself is set to a large number to exclude itself from the nearest
      cdist = (cdist + (nrobots - r1));
    }

    // update the matrix of nearest robots
    rbd = robotsbydistance;
    for (r1=0, cdist = robotsdist; r1 < nrobots; r1++, cdist = (cdist + nrobots))
    {
      remaining = (nrobots - 1);
      for (r2=0; r2 < (nrobots - 1); r2++)
      {
        for (r3=0, ccdist = cdist, smallerid=0, smallerdist = *ccdist, rbddist = ccdist; r3 < nrobots; r3++, ccdist++)
        {
          if (*ccdist <= smallerdist)
          {
            smallerdist = *ccdist;
            smallerid = r3;
            rbddist = ccdist;
          }
        }
        if (smallerdist < 1000000)  // we ignore robots located at a power dstance greater that 750*750
         {
          *rbd = smallerid;
          rbd++;
          *rbddist = 9999999999.0;
          remaining--;
         }
         else
         {
          *rbd = -1;
          rbd = (rbd + remaining);
          break;
         }
      }
      *rbd = -1;  // we use -1 to indicate the end of the list
      rbd++;

    }
}

/*
 *  omnidirectional camera that perceives the color blobs constituted by the other robots
 *  robots can have their color leds turned on in red or blue
 *  it update four sensors that encode the fraction of red and blue pixels detected in the two frontal sectors of the camera
 */

int initCameraSensorRFB(struct robot *cro)
{

   int i;
   double  **camb;
   int *nb;

   cro->camnsectors = 2;

   if (cro->idn == 0)
     printf("Sensor[%d]: camera2, %d sectors, %d colors \n", cro->camnsectors * 2, cro->camnsectors, 2);

   cro->camblobsn = (int *) malloc(cro->camnsectors * sizeof(double));
   // allocate space and initialize
   cro->camblobs = (double **) malloc(cro->camnsectors * sizeof(double *));
   for (i=0, camb = cro->camblobs, nb = cro->camblobsn; i < cro->camnsectors; i++, camb++, nb++)
     {
      *camb = (double *)malloc(nrobots*4*8 * sizeof(double));
      *nb = 0;
     }
   return(cro->camnsectors * 2);
}

/*
 * add a new blob to the blob list of a sector
 * providing that it does not overlap with previously stored blobs
 * it receive as input the pointer to the first blob of the sector-list, the number of existing blobs, the blob color, and the start and ending angle
 * assume that the starting and ending angles are in the range [0, PI2]
 * blobs smaller than the resolution of the camera (0.1 degrees, 0.00174 radiants) are filtered out
 */
void updateCameraAddBlob(double *cb, int *nb, double color, double dist, double brangel, double branger)

{

   int b;

   // we ignore small blobs with a negative intervals since they are spurious
   if ((branger - brangel) < 0.00174)
   {
      return;
   }

   // check whether this blob overlap with preexisting ones
   for (b=0; b < *nb; b++)
   {
       cb++;
       cb++;
       // if fully overlap with previous blobs we simply filter it out
       if (anginrange(brangel, *cb, *(cb + 1)) && anginrange(branger, *cb, *(cb + 1)))
         {
          return;
         }
         else
         {
         // if start inside an existimg blob but ends after the end of the existing blobs we trim the first part of the blob
         if (anginrange(brangel, *cb, *(cb + 1)))
           {
             brangel = *(cb + 1);
           }
           else
            {
              // if end inside an existing blob but starts outside the existing blob we trim the last part of the blob
               if (anginrange(branger, *cb, *(cb + 1)))
                   {
                     branger = *cb;
                   }
            }
         }
         cb++;
         cb++;
   }

   // we ignore small blobs with a negative intervals since they are spurious
   // the blob could had become too small after being trimmed
   if ((branger - brangel) < 0.00174)
   {
      return;
   }

   *cb = color; cb++;
   *cb = dist; cb++;
   *cb = brangel; cb++;
   *cb = branger; cb++;
   *nb += 1;

}


void updateCameraSensorRFB(struct robot *cro, int *rbd, int noutputs)

{


	int s;						// sector
    struct robot *ro;			// pointer to robot list
	int r;						// robot id
    double v1, v2;				// visible arc of the robot (from angle v1 to angle v2)
	double x1,x2,x3,y1,y2,y3;   // the coordinate of the initial and final points of the two blobs
	double a1,a2,a3;			// angle of the initial and final points of the two adjacent blobs
	double ab1, ab2;			// the angular boundaries between the frontal and rear side
	double ab;					// the angular boundary located within the visible arc
	double d1, d2;				// distance of the two frontal/rear borders
	double ab1x, ab1y, ab2x,ab2y; // coordinates of the two borders
	int ac;						// selected front/rear border
	double ang;					// the angle of the perceiving robot from the perceived robot
	double dist2;               // the power distance between the two robots
	double rangel, ranger;		// initial and final angle of the current sector
	double cblob[3][2];			// color blobs ([Red, Blue, Green][ang-start, ang-end])
	double **camb;              // pointer to blob matrix
	double *cb;					// pointer to blobs of a sectors
	int *nb;					// pointer to the number of blobs x sectors
    double act[10];				// activation of the current visual sector (0=red, 1=blue)
	double secta;				// amplitude of visual sectors
	int c, color;				// color
    double bcolor;              // blob color
    double buf;
	int b;
	double out[100];
	int i;

	color = -1;
	
    secta = M_PI / 3.0; //PI2 / (double) cro->camnsectors; // / 3.0;        // angular amplitude of the camera sectors
    for(s=0, nb = cro->camblobsn; s < cro->camnsectors; s++, nb++)
	  *nb = 0;
	// we extract a red or blue color blob for each perceived robot
	// we stored the visible blobs divided by visual sectors and color
	// finally we compute the fraction of pixels for each sector and each color
    for (r=0; r < nrobots; r++)
	    {
          if (rbd[r] < 0)     // if the list of nearby robots ended we exit from the for
             break;
          ro=(rob + rbd[r]);  // the r nearest perceived robot

	// Convert output activations in the proper ranges when tanh is used
	if (afunction == 2)
	{
		for (i = 0; i < noutputs; i++)
		{
			out[i] = caction[rbd[r] * noutputs + i];
			out[i] = 1.0 + ((out[i] - 1.0) / 2.0);
		}
	}
	else
	{
		for (i = 0; i < noutputs; i++)
			out[i] = caction[rbd[r] * noutputs + i];
	}
		

          if (1 > 0 /* cro->idn != ro->idn*/)
		   {
			// angle from perceived to perceiving robot
			ang = angv(cro->x, cro->y, ro->x, ro->y);
			// compute the visibile and coloured angular intervals
			v1 = ang - (M_PI / 2.0);
			v2 = ang + (M_PI / 2.0);
			ab1 = ro->dir - (M_PI / 2.0);
			ab2 = ro->dir + (M_PI / 2.0);
			// identify the relevant boundary (the boundary that is visible from the point of view of the perceiving robot)
			// we do that by checking the point that is nearer from the perceiving robot
			ab1x = ro->x + xvect(ab1, ro->radius);
			ab1y = ro->y + yvect(ab1, ro->radius);
			ab2x = ro->x + xvect(ab2, ro->radius);
			ab2y = ro->y + yvect(ab2, ro->radius);
			d1 =((ab1x - cro->x)*(ab1x - cro->x) + (ab1y - cro->y)*(ab1y - cro->y));
			d2 =((ab2x - cro->x)*(ab2x - cro->x) + (ab2y - cro->y)*(ab2y - cro->y));
			// the left and right border are followed and preceeded by different colors
			if (d1 <= d2)
			  {
			   ab = ab1;
			   ac = 0;
			   }
			  else
			  {
			   ab = ab2;
			   ac = 1;
			  }
			// calculate the xy coordibate of the three points located on the borders of the perceived robot
			x1 = ro->x + xvect(v2, ro->radius);
			y1 = ro->y + yvect(v2, ro->radius);
			x2 = ro->x + xvect(ab, ro->radius);
			y2 = ro->y + yvect(ab, ro->radius);
			x3 = ro->x + xvect(v1, ro->radius);
			y3 = ro->y + yvect(v1, ro->radius);
			// calculate the correspoding three angle from the point of view of the perceiving robot
			a1 = angv(x1, y1, cro->x, cro->y);
			a2 = angv(x2, y2, cro->x, cro->y);
			a3 = angv(x3, y3, cro->x, cro->y);
			// extract the angular intervals of the red and blue subsections
			if (ac == 0)
			 {
			  cblob[0][0] = a1;
			  cblob[0][1] = a1 + angdelta(a1, a2);
			  cblob[1][0] = a3 - angdelta(a2, a3);
			  cblob[1][1] = a3;
			 }
			 else
			 {
			  cblob[1][0] = a1;
			  cblob[1][1] = a1 + angdelta(a1, a2);
			  cblob[0][0] = a3 - angdelta(a2, a3);
			  cblob[0][1] = a3;
			 }

            // angles sanity checks
            for (c=0; c < 2; c++)
            {
              // if the first angle is negative the blog is over the border
              // we make both angle positive
              // it will the be divided in two blobs below because the ending angle will exceed PI2
              if (cblob[c][0] < 0)
                   {
                      cblob[c][0] += PI2;
                      cblob[c][1] += PI2;
                   }
              // if the second angle is smaller than the first and the interval is small, we invert them
              // apparently this is due to limited precision of angle calculation
              if ((cblob[c][1] - cblob[c][0]) < 0)
                 {
                    buf = cblob[c][0];
                    cblob[c][0] = cblob[c][1];
                    cblob[c][1] = buf;
                 }
             }

            /*
            for (c=0; c < 2; c++)
            {
              if ((cblob[c][1] - cblob[c][0]) < 0)
                 {
                    printf("negative %.4f %.4f   %.4f %d ", cblob[c][0], cblob[c][1], cblob[c][1] - cblob[c][0], ac);
                    if (ac == 0 && c == 0) printf("red  (%.4f %.4f %.4f) a1 a2 %.4f %.4f a1 + a1_a2 %.4f \n", a1, a2, a3, a1, a2, angdelta(a1, a2));
                    if (ac == 0 && c == 1) printf("blue (%.4f %.4f %.4f) a3 a2 %.4f %.4f a3 - a2_a3 %.4f\n", a1, a2, a3, a3, a2, angdelta(a2, a3));
                    if (ac == 1 && c == 1) printf("blue (%.4f %.4f %.4f) a1 a2 %.4f %.4f a1 + a1_a2 %.4f \n", a1, a2, a3, a1, a2, angdelta(a1, a2));
                    if (ac == 1 && c == 0) printf("red  (%.4f %.4f %.4f) a3 a2 %.4f %.4f a3 - a2_a3 %.4f \n", a1, a2, a3, a3, a2, angdelta(a2, a3));
                 }

              if ((cblob[c][1] - cblob[c][0]) > 0.8)
                 printf("large %.4f %.4f   %.4f %d\n", cblob[c][0], cblob[c][1], cblob[c][1] - cblob[c][0], ac);
            }
            */

            // we store the two blobs
            // blobs extending over PI2 are divided in two
            dist2 =((ro->x - cro->x)*(ro->x - cro->x) + (ro->y - cro->y)*(ro->y - cro->y));
            camb = cro->camblobs;
            nb = cro->camblobsn;
            cb = *camb;
            // we check whether frontal red leds are turned on or not
            if (ro->motorleds == 0 || out[2] > 0.5)
               bcolor = 1.0;
              else
               bcolor = 0.0;
            if (cblob[0][1] < PI2)
            {
              updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[0][0], cblob[0][1]);
            }
            else
            {
              updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[0][0], PI2);
              updateCameraAddBlob(cb, nb, bcolor, dist2, 0.0, cblob[0][1] - PI2);
            }
            // we check whether rear blue leds are turned on or not
            if (ro->motorleds == 0 || out[3] > 0.5)
               bcolor = 2.0;
              else
               bcolor = 0.0;
            if (cblob[1][1] < PI2)
            {
              updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[1][0], cblob[1][1]);
            }
            else
            {
              updateCameraAddBlob(cb, nb, bcolor, dist2, cblob[1][0], PI2);
              updateCameraAddBlob(cb, nb, bcolor, dist2, 0.0, cblob[1][1] - PI2);
            }

		  }  // end if (cro->idn != ro->idn)
		}  // end for nrobots



    // sum the angular contribution of each relevant blob to each color sector
    double inrange;
    double addsrangel;  // additional sector rangel
    double addsranger;  // additional sector ranger
    int addsid;         // sector to which the additial sector belong
    int ss;             // id of the sector, usually = s, but differ for the additional sector
    double *cbadd;      // pointer to blob list used to add a new sub-blob

    // initialize to 0 neurons actiovation
    for(b=0; b < cro->camnsectors * 2; b++)
       act[b] = 0.0;

    camb = cro->camblobs;
    cb = *camb;
    nb = cro->camblobsn;
    b = 0;
    while (b < *nb)
     {
       inrange=false;
       addsid = -1;  // the id of the additional sensors is initialized to a negative number
       if (*cb == 0.0) color = -1; // black
       if (*cb == 1.0) color = 0; // red
       if (*cb == 2.0) color = 1; // blue
       //if (cro->idn == 0) printf("b %d) %.2f %.2f %.2f %.2f (%.2f) \n", b, *cb, *(cb + 1), *(cb + 2), *(cb + 3), *(cb + 3) - *(cb + 2));
       for(s=0, rangel = cro->dir - secta, ranger = rangel + secta; s < (cro->camnsectors + 1); s++, rangel += secta, ranger += secta)
         {

           if (s < cro->camnsectors)
           {
            ss = s;
            //if (cro->idn == 0) printf("sector %d (ss %d) %.2f %.2f  \n", s, ss, rangel, ranger);
            // we normalize the angles of the sector in the range [0, PI2+sectora]
            if (rangel < 0.0)
            {
             rangel += PI2;
             ranger += PI2;
            }
            // if the current sector extend over PI2 we trim it to PI2 and we initialize the additional sector
            if (ranger > PI2)
            {
              addsrangel = 0.0;
              addsranger = ranger - PI2;
              addsid=s;
            }
           }
           else
           {
            // if an additional sensor has been defined we process is otherwise we exit from the sector for
            if (addsid >= 0)
            {
              ss = addsid;
              // if (cro->idn == 1) printf(" Additional sector s %d ss %d addsid %d range %.2f %.2f\n", s, ss, addsid, addsrangel, addsranger);
              rangel = addsrangel;
              ranger = addsranger;
            }
           else
            {
             break;
            }
           }
           //if (cro->idn == 0) printf("sector %d (ss %d) %.2f %.2f  \n", s, ss, rangel, ranger);
           if (color >= 0)
           {
            if ((*(cb + 2) >= rangel) && (*(cb + 2) < ranger) && (*(cb + 3) >= rangel) && (*(cb + 3) < ranger) ) // coloured blob fully contained in the sector
             {
              act[ss * cro->camnsectors + color] += *(cb + 3) - *(cb + 2);
              //if (cro->idn == 1) printf("fullin rodir %.2f sector %d %.2f %.2f blobcolor %.2f blobang %.2f %.2f (%.2f)\n", cro->dir, s, rangel, ranger, *cb, *(cb + 2), *(cb + 3), *(cb + 3) - *(cb + 2));
              inrange=true;
             }
            else
             {
              if ((*(cb + 2) >= rangel) && (*(cb + 2) < ranger) && (*(cb + 3) >= rangel))  // non-black blob staring inside and ending outside, inside the next sector
                {
                  act[ss * cro->camnsectors + color] += ranger - *(cb + 2);
                  // we use the exceeding part to create a new blob added at the end of the blob list
                  camb = cro->camblobs;
                  cbadd = *camb;
                  cbadd = (cbadd + (*nb * 4));
                  *cbadd = *cb; cbadd++;
                  *cbadd = *(cb + 1); cbadd++;
                  *cbadd = ranger; cbadd++; // the new blob start from the end of the current sector
                  *cbadd = *(cb + 3); cbadd++;
                  *nb = *nb + 1;
                  //printf("added blob %d %.2f %.2f %.6f %.6f  range %.6f %.6f \n", *nb, *cb, *(cb + 1), ranger, *(cb + 3), rangel, ranger);
                  //if (cro->idn == 1) printf("startin rodir %.2f sector %d %.2f %.2f blobcolor %.2f blobang %.2f %.2f (%.2f)\n", cro->dir, s, rangel, ranger, *cb, *(cb + 2), ranger, ranger - *(cb + 2));
                  inrange=true;
                }

             }
           }
         }
        if (!inrange)   // blobs outsiode the view range of all sectors are turned in black for graphical purpose
           *cb = 0.0;

        cb = (cb + 4);
        b++;
     }
    //if (cro->idn == 0) printf("\n\n");


    // we finally store the value in the input units
    //printf("activation s1 red blue s2 red blue robot %d ", cro->idn);
    for(b=0; b < cro->camnsectors * 2; b++, cro->csensors++)
    {
       *cro->csensors = act[b];
       //printf("%.2f ", act[b]);
    }
    //printf("\n");

 
}

/*
 *    update the state of the ground sensor on circular target areas
 *    the first neuron is turned on when the robot is on a target area with a color < 0.25
 *    the second neuron is turned on when the robot is on a target area with a color > 0.25 and < 0.75
 */
int initGroundSensor(struct robot *cro)
{
  if (cro->idn == 0) printf("Sensor[%d]: ground color \n", 2);
  return(2);
}

void updateGroundSensor(struct robot *cro)
{

	int o;
	double dx, dy, cdist;
    double act[2];
	
	act[0] = act[1] = 0.0;
	
	for (o=0; o < nenvobjs; o++)
	 {
	  if (envobjs[o].type == STARGETAREA)
	  {
	    dx = cro->x - envobjs[o].x;
		dy = cro->y - envobjs[o].y;
		cdist = sqrt((dx*dx)+(dy*dy));
		if (cdist < envobjs[o].r)
			{
			  if (envobjs[o].color[0] < 0.25)
			    act[0] = 1.0;
			   else
			   {
				if (envobjs[o].color[0] < 0.75)
			            act[1] = 1.0;
			   }
			 }
		}
	  }
    *cro->csensors = act[0];
    cro->csensors++;
    *cro->csensors = act[1];
    cro->csensors++;
}


