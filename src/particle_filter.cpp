/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::default_random_engine gen_;

	num_particles = 100;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i=0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen_);
		particle.y = dist_y(gen_);
		particle.theta = dist_theta(gen_);
		particle.weight = 1;
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	std::default_random_engine gen_;

	float dt = delta_t;
	float v = velocity;
	float th_dot = yaw_rate;

	for (int i=0; i < num_particles; i++) {
		Particle particle = particles[i];
		double x = particle.x;
		double y = particle.y;
		double th = particle.theta;
        
        double x_p, y_p;
        if (fabs(th_dot) < 0.001) {
            x_p = x + v * dt * cos(th);
            y_p = y + v * dt * sin(th);
        } else {
            x_p = x + v / th_dot * (sin(th + th_dot * dt) - sin(th));
            y_p = y + v / th_dot * (cos(th) - cos(th + th_dot * dt));
        }
		double th_p = th + th_dot * dt;
        
		normal_distribution<double> dist_x(x_p, std_pos[0]);
		normal_distribution<double> dist_y(y_p, std_pos[1]);
		normal_distribution<double> dist_theta(th_p, std_pos[2]);

		particle.x = dist_x(gen_);
		particle.y = dist_y(gen_);
		particle.theta = dist_theta(gen_);

		particles[i] = particle;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (int i=0; i < observations.size(); i++) {
		float min_distance = std::numeric_limits<float>::max();
		int min_index;
		LandmarkObs obs_a = observations[i];
		for (int j=0; j < predicted.size(); j++) {
			LandmarkObs obs_b = predicted[j];
			float distance = sqrt( pow(obs_b.x - obs_a.x, 2.0) + pow(obs_b.y - obs_a.y, 2.0) );
			if (distance < min_distance) {
				min_distance = distance;
				min_index = j;
			}
		}
		observations[i].id = predicted[min_index].id;
		predicted.erase(predicted.begin() + min_index);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// observations is in particle (vehicle) coordinate system
	// map_landmarks is in map coordinate system
	float s_x = std_landmark[0];
	float s_y = std_landmark[1];

	for (int i=0; i < num_particles; i++) {
		Particle p = particles[i];

		std::vector<LandmarkObs> predicted_landmark_positions;
		for (int j=0; j < map_landmarks.landmark_list.size(); j++) {
			LandmarkObs p_o;
			float x_map = map_landmarks.landmark_list[j].x_f;
			float y_map = map_landmarks.landmark_list[j].y_f;
			p_o.id = map_landmarks.landmark_list[j].id_i;

			// Translate then rotate to get map landmark in map space
			// to be in vehicle coordinate space from perspective of our particle
			p_o.x = (x_map - p.x) * cos(-p.theta) - (y_map - p.y) * sin(-p.theta);
			p_o.y = (x_map - p.x) * sin(-p.theta) + (y_map - p.y) * cos(-p.theta);
			// predicted_landmark_positions in particle vehicle coordinate space
			predicted_landmark_positions.push_back(p_o);
		}
		// Assigns id values to LandmarkObs in observations variable, passed by reference
		ParticleFilter::dataAssociation(predicted_landmark_positions, observations);

		double w = 1;
		for (int j=0; j < observations.size(); j++) {
			LandmarkObs a = observations[j];
			LandmarkObs b;
			for (int k=0; k < predicted_landmark_positions.size(); k++) {
				if (predicted_landmark_positions[k].id == a.id) {
					b = predicted_landmark_positions[k];
					break;
				}
			}
			double c = (1 / (2 * M_PI * s_x * s_y));
			double x_part = pow(a.x - b.x, 2.0) / (2 * s_x * s_x);
			double y_part = pow(a.y - b.y, 2.0) / (2 * s_y * s_y);
			w *= c * exp(-(x_part + y_part));
		}
		
		p.weight = w;
		particles[i] = p;
	}

	float sum = 0;
	for (int i=0; i < num_particles; i++) {
		sum += particles[i].weight;
	}
	for (int i=0; i < num_particles; i++) {
		particles[i].weight /= sum;
	}
}

void ParticleFilter::resample() {
	std::default_random_engine gen_;

	std::vector<float> weights;
	for (int i=0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	std::discrete_distribution<int> distribution(weights.begin(), weights.end());

	std::vector<Particle> new_particles;
	while (new_particles.size() < num_particles) {
		int idx = distribution(gen_);
		new_particles.push_back(particles[idx]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
