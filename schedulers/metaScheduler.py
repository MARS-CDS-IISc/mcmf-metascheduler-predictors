#!/usr/bin/env python2.4

from base.prototype import JobSubmissionEvent, JobTerminationEvent, JobPredictionIsOverEvent, JobStartEvent
from base.prototype import ValidatingMachine
from base.event_queue import EventQueue
from common import CpuSnapshot, list_print

from easy_plus_plus_scheduler import EasyPlusPlusScheduler
from shrinking_easy_scheduler import ShrinkingEasyScheduler

import networkx as nx

from simulator import Simulator

from predictors.wait import QueueWaitingTime
from predictors.electricity_price import predictPrice
import cPickle as pickle
from random import randint
WaitPredictor = QueueWaitingTime.Batch_System
ElectricityPricePredictor = predictPrice.PricePredictor

import math
import sys
import random

EnabledWaitPred = True
EnabledPricePred = True

class MetaSchedulerEvent(object):
    global_event_counter = 0
    @classmethod
    def next_counter(cls):
        cls.global_event_counter += 1
        return cls.global_event_counter

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.counter = MetaSchedulerEvent.next_counter()

    def __repr__(self):
        return type(self).__name__ + "<timestamp=%(timestamp)s>" % vars(self)

    def __cmp__(self, other):
        return cmp(self._cmp_tuple, other._cmp_tuple)

    @property
    def _cmp_tuple(self):
        "Order by timestamp. A global counter tie-breaks."
        return (self.timestamp, self.counter)

    def __eq__(self, other):
        return self._eq_tuple == other._eq_tuple

    @property
    def _eq_tuple(self):
        "equal iff timestamp, type are the same"
        return (self.timestamp, type(self))

class TriggerMetaScheduler(MetaSchedulerEvent):pass
class TriggerMetrics(MetaSchedulerEvent):pass

class JobHolder(object):
    def __init__(self, event_queue, cycleLength=2*60):
        self.heldJobs = []
        self.heldEvents = []
        self.heldJobID = []
        self.firstJobTS = 0
        self.SchedulingCycleLength = cycleLength # in seconds
        self.event_queue = event_queue
        self.onlyOnce = True
    def addNewJob(self, njob, nevent):

        if self.heldJobs == []:
            self.firstJobTS = nevent.job.submit_time
            self.event_queue.add_event(TriggerMetaScheduler(timestamp=self.firstJobTS+self.SchedulingCycleLength))
            if self.onlyOnce:
                self.event_queue.add_event(TriggerMetrics(timestamp=self.firstJobTS+3600))
                self.onlyOnce = False
        self.heldJobs.append(njob)
        self.heldJobID.append(njob.id)
        self.heldEvents.append(nevent)

        '''
        if (njob.submit_time - self.firstJobTS) >= self.SchedulingCycleLength:
            print 'Invoking MCMF from JobHolder - elapsed time is', (njob.submit_time - self.firstJobTS)/60.0, 'minutes'
            return True
        else:
            return False
        '''
    def delJob(self, job):
        jobIndex = self.heldJobID.index(job.id)
        del self.heldJobs[jobIndex] # TODO: possible bug: need to correct the value of first job TS
        del self.heldJobID[jobIndex]
        del self.heldEvents[jobIndex]
        
    def clearJobs(self):
        self.heldJobs = []

    def resetAllSubmitTimes(self, current_time):
        for job in self.heldJobs:
            job.submit_time = current_time
        for event in self.heldEvents:
            event.timestamp = current_time

    def printJobDetails(self):
        print 'Held ID, Submit'
        for job in self.heldJobs:
            print job.id, job.submit_time
        print ""
    
    def countWaitingJobs(self):
        return len(self.heldJobs)

'''
* This class is the common interface between the User, Predictor and Site Scheduler
* Tasks:
*  Accept job inputs from user
*  Obtain predictions
*  Submit jobs to scheduler
*  Provide feedback (if necessary) to user and predictor
'''
class MetaScheduler(object):

    def __init__(self, inputConfiguration, jobs, alpha, beta, algorithm, joblimit,maxq, pfile, mperc, pricepath):
        #print 'Configuring meta-scheduler  with', 'Jobs:', jobs, 'input configuration:', inputConfiguration
        self.event_queue = EventQueue() # This is a global event queue for all the sites
        self.SchedulingCycleLength = 60

	self.pickleFile = pfile
	self.mperc = mperc
        self.time_of_last_job_submission = 0
        self.currentSubmitCount = 0
        self.jobHoldingLayer = JobHolder(self.event_queue, self.SchedulingCycleLength)
        self.jobs = jobs
        self.all_terminated_jobs=[]
        self.qwait_error_count = 0
        self.scheduling_cycle_count = 0

        # Global start and end times
        self.target_job_start_time = 0 
        self.target_job_end_time = 0
        self.historySetSize = 500*len(inputConfiguration)      # TODO: Require this much history at all sites together
        #print 'History size total:', self.historySetSize
        self.begin_halt = False

        self.alphaValue = alpha
        self.betaValue = beta
        self.strategy = algorithm
        
        self.processed_target_jobs = 0
        self.max_target_jobs = joblimit
        if joblimit == 0:
            self.max_target_jobs = float('inf')

        self.MaxQ = maxq
        self.event_queue.add_handler(JobSubmissionEvent, self.handle_submission_event)
        self.event_queue.add_handler(JobTerminationEvent, self.handle_termination_event)
        self.event_queue.add_handler(JobStartEvent, self.handle_start_event)
        self.event_queue.add_handler(TriggerMetaScheduler, self.invoke_matchmaker)
        self.event_queue.add_handler(TriggerMetrics, self.compute_instant_metrics)

        #if isinstance(scheduler, EasyPlusPlusScheduler) or isinstance(scheduler, ShrinkingEasyScheduler):
        #    self.event_queue.add_handler(JobPredictionIsOverEvent, self.handle_prediction_event)
        
        countSubmissions = 0    
        for job in self.jobs:
            countSubmissions += 1
            self.event_queue.add_event( JobSubmissionEvent(job.submit_time, job) )     

        #if countSubmissions % 1000 == 0:
        #    print '** Added', countSubmissions, 'job submission events to event queue **'

        '''
        * Setting up sites, their predictors etc. 
        * Input configuration is a set of locations and their attributes
        '''
        self.siteSubmitCount = []
        self.currentSite = 0 # For history creation 
        self.siteInfo = inputConfiguration
        self.siteIDMap = {}
	self.MaxGridProcs = 0
	self.MaxSiteProcs = []
	self.scoreList = []
        print 'Setting up sites:'
        for site in self.siteInfo:
            self.siteIDMap[site.id] = site

            waitTimePredictor = None
            if EnabledWaitPred:
                waitTimePredictor = WaitPredictor(1, site.max_ert, site.max_req)              
            site.wait_predictor = waitTimePredictor
            site.price_predictor = None
            if EnabledPricePred:
                site.price_predictor = ElectricityPricePredictor(site.location)
                site.price_predictor.load_data(pricepath, site.location + '.csv')
                print 'Loading', site.location + '.csv', 'for', site.name
		self.scoreList.append([(site.price_predictor.get_average_price())*site.power_per_core, site.id])
            site.machine = ValidatingMachine(num_processors=site.num_processors, 
                                             event_queue=self.event_queue)

            site.queueSimulator = Simulator(jobs, 
                                            site.num_processors, 
                                            site.scheduler, 
                                            self.event_queue, 
                                            site.machine, )
            self.siteSubmitCount.append(0)
            self.MaxGridProcs += site.num_processors 
            self.MaxSiteProcs.append(site.num_processors)
	#for item in sorted(self.scoreList):
	#	print item
	#print ""
	#for item in sorted(self.scoreList):
	#	print item[1],',',
	#print ""
        #for site in self.siteInfo:
        #    print site.id, site.queueSimulator, site.machine, site.wait_predictor, site.num_processors
        
        # Initializing stat counters 
        self.total_wait_time = 0
        self.total_meta_wait_time = 0
        self.total_response_time = 0
        self.total_bounded_slowdown = 0
        self.total_electricity_price = 0
        self.total_jobs_for_stats = 0
	self.utilization_num = 0 
	self.first_job_submit = float('inf')
        self.last_job_finish = -float('inf')

        self.site_wait_time = [0]*len(self.siteInfo)
        self.site_response_time = [0]*len(self.siteInfo)
        self.site_bounded_slowdown = [0]*len(self.siteInfo)
        self.site_electricity_price = [0]*len(self.siteInfo)
        self.site_jobs_for_stats = [0]*len(self.siteInfo)
        self.site_utilization_num = [0]*len(self.siteInfo)
        self.site_first_job_submit = [float('inf')]*len(self.siteInfo)
        self.site_last_job_finish = [-float('inf')]*len(self.siteInfo)

	self.GSL = [] # GridSchedulingLog

        self.instLoad = []
        self.instPower = []
        self.instPrice = []
        self.instHour = []

        self.total_meta_jobs = 0
        self.pricepath = pricepath

    def compute_instant_metrics(self, event):
	return
        if self.currentSubmitCount <= self.historySetSize:
            return
        if self.begin_halt:
            return
        # For each site, get the current load = (sum of remaining CPU hours of running jobs + sum of CPU hours of queued jobs)/MaxCPU
        current_time = event.timestamp
        load = []
        for site in self.siteInfo:
            queuedHours = 0
            for job in site.queuedJobs:
                queuedHours += job.num_required_processors*job.user_estimated_run_time
            for job in site.runningJobs:
                queuedHours += job.num_required_processors*(current_time - job.start_to_run_at_time)
            queuedHours = float(queuedHours)/(site.num_processors * site.max_ert)
            load.append(queuedHours)
        self.instLoad.append(load)
        power = []
        for site in self.siteInfo:
            p = 0
            for job in site.runningJobs:
                p += job.get_watts(site.power_per_core)
            power.append(p)
        self.instPower.append(power)
        cost = []
        for site in self.siteInfo:
            cost.append(site.price_predictor.get_current_price(current_time))
        self.instPrice.append(cost)
        self.instHour.append(current_time/(60*60))
        self.event_queue.add_event(TriggerMetrics(timestamp=current_time+3600))

    def handle_submission_event(self, event):
        '''
        * Some points to check:
        * - At the end of submission to a site, make sure the wait time predictor has the correct job config 
        *   and the correct number of submitted jobs, remove tentative arrivals
        * - 
        '''
        assert isinstance(event, JobSubmissionEvent)
        
        self.currentSubmitCount += 1
       
        '''
        * Create enough history at each sie
        '''
        event.job.origin_site = self.siteInfo[event.job.origin_site_id - 1]
        if self.currentSubmitCount <= self.historySetSize:
            # Need a fix here to ensure that a job is submited only to a compatible site
            self.currentSite = event.job.origin_site_id - 1
            '''
            for i in range(len(self.siteInfo)):
                s = (self.currentSite + i) % len(self.siteInfo)
                if self.siteInfo[s].num_processors > event.job.num_required_processors:
                    break
                   
            hSite = self.siteInfo[((self.currentSite + i) % len(self.siteInfo))]
            if event.job.id > self.historySetSize - 25:
                print 'HC:Submitting job', event.job.id, 'to site', hSite.id
                print self.siteSubmitCount
            '''
            hSite = self.siteInfo[self.currentSite]
            
            event.job.siteid = hSite.id
            event.job.sitemap = hSite
            event.job.target_site = hSite # Origin site = Target site for history jobs
            hSite.queuedJobs.append(event.job)

            hSite.wait_predictor.notify_arrival_event(event.job, hSite.queuedJobs, hSite.runningJobs)
            hSite.queueSimulator.handle_submission_event(event)
            if event.job.id % 1000 == 0:
                print 'Finished', event.job.id, 'job submissions for history'
            for site in self.siteInfo:
                last_time = event.timestamp - 3*24*60*60
                last_time_midnight = int(last_time/86400)*86400
                site.price_predictor.initialize_start_time(max(last_time_midnight + site.timezone_lag*60*60, 0))
            self.target_job_start_time = event.timestamp
            self.target_job_end_time = self.target_job_start_time + 10*365*24*60*60 # assumption - max simulation length of 10 years

            # If we go on initializing in this manner, each site will have the the last history job's timestamp as start time of target jobs - good enough for our purpose - once target jobs start this value will remain fixed and this is same for all sites.
            self.siteSubmitCount[self.currentSite] += 1
            
            #self.currentSite = (1 + self.currentSite)%len(self.siteInfo)
        
            return

        self.processed_target_jobs += 1
          
        '''
        * For each submission, send it to a holding layer which invokes MCMF at intervals of 5 mins (virtual time)
        '''
        
        #print self.siteSubmitCount

        if (event.timestamp < self.target_job_end_time and (not self.begin_halt)) and (self.processed_target_jobs < self.max_target_jobs):
            #print '*****DEBUG MSG*******', self.processed_target_jobs, self.max_target_jobs, self.mperc, self.strategy
	    select = randint(1, 100)
	    do_baseline = False
            if select > self.mperc:
            	do_baseline = True 
            if self.strategy == 9 or do_baseline:
                self.currentSite = event.job.origin_site_id - 1
                hSite = self.siteInfo[self.currentSite]
                event.job.siteid = hSite.id
                event.job.sitemap = hSite
                event.job.target_site = hSite # Origin site = Target site for history jobs
                hSite.queuedJobs.append(event.job)
                hSite.wait_predictor.notify_arrival_event(event.job, hSite.queuedJobs, hSite.runningJobs)
                hSite.queueSimulator.handle_submission_event(event)
            elif self.strategy == 1 or self.strategy == 2 or self.strategy == 3:
                #print 'Adding job to holding layer', 'ID:', event.job.id, 'Submit:', event.job.submit_time, 'Held:',len(self.jobHoldingLayer.heldJobs),'jobs'
                self.jobHoldingLayer.addNewJob(event.job, event)
                event.job.isMeta = True
		self.total_meta_jobs += 1
        else:
            self.begin_halt = True

    '''
    * Core Min Cost Max Flow based matchmaker
    * Note: One way is to iteratively do the below till all jobs are scheduled or machine capactities are exhausted. 
    * TODO: Defer remaining jobs to next scheduling cycle. 
    '''
    def invoke_matchmaker(self, event):
        assert isinstance(event, TriggerMetaScheduler)
        
        self.scheduling_cycle_count += 1
        print 'Scheduling cycle', self.scheduling_cycle_count
        # 0. Now all events are frozen. Only after this scheduling cycle other events are processed.
        # 0. Reset the submit time of jobs in holding layer to current time 
        
        #print 'Invoking matchmaker - resetting submit times for held jobs'

        self.current_time = event.timestamp 
        self.jobHoldingLayer.resetAllSubmitTimes(self.current_time)
        #self.jobHoldingLayer.printJobDetails()
        
        # 1. Invoke predictors for wait time and price and find costs for each job on each site
        assignmentCost, wait_time_matrix, response_time_matrix, power_price_matrix, rejectJobs  = self.compute_scheduling_cost()
        
        if assignmentCost == None:
            return None

        # 2. Construct flow network
        jobSiteFlowNetwork, source, sink, jMap, sMap = self.create_flow_network(assignmentCost)
        
        # 3. Invoke Minimum Cost Maximum Flow
        flowValues = nx.max_flow_min_cost(jobSiteFlowNetwork, source, sink)
        
        # 4. Output of MCMF to decide appropriate job submissions to the queues
        submissionList = self.post_process_flow(flowValues, jMap, sMap, rejectJobs, assignmentCost)
        
	# 5. Submit jobs
        #print 'Submission list:', submissionList
        for item in submissionList:
            job_id = item[0]
            event = self.jobHoldingLayer.heldEvents[self.jobHoldingLayer.heldJobID.index(job_id)]
            site = self.siteIDMap[item[1]]
            #print 'Actual grid: Submitting job', event.job.id, 'to site', site.id
            event.job.siteid = site.id
            event.job.sitemap = site
            event.job.target_site = site
            site.queuedJobs.append(event.job)
            # Modify job's runtime and ERT, scale it to the submitted system
            event.job.user_estimated_run_time = max(int(event.job.original_ert*float(event.job.origin_site.gflops/event.job.target_site.gflops)), 1)
            event.job.actual_run_time = max(int(event.job.original_run*float(event.job.origin_site.gflops/event.job.target_site.gflops)), 1)
            event.job.predicted_run_time = event.job.user_estimated_run_time
            if job_id in wait_time_matrix.keys():
                event.job.wait_time_prediction = wait_time_matrix[job_id][site.id]
                event.job.response_time_prediction = response_time_matrix[job_id][site.id]
            site.wait_predictor.notify_arrival_event(event.job, site.queuedJobs, site.runningJobs)
            site.queueSimulator.handle_submission_event(event)
            self.jobHoldingLayer.delJob(event.job)

        if self.jobHoldingLayer.countWaitingJobs():
            self.event_queue.add_event(TriggerMetaScheduler(timestamp=event.timestamp+self.SchedulingCycleLength))         

    def compute_scheduling_cost(self):
        self.assign_condition_matrix = {}
        wait_time_matrix = {}
        response_time_matrix = {}
        predicted_end_time_matrix = {}
        min_resp = float('inf')
        max_resp = -float('inf')
        for site in self.siteInfo:
            site.max_resp_in_cycle = -float('inf')
        power_price_matrix = {}
        min_price = float('inf')
        max_price = -float('inf')
        alpha = self.alphaValue
        beta = self.betaValue
        rejectJobs = set([])
        
        # TODO: Add condition about max.ert also - i.e. for a job, scaled ERT should be less than max ert possible at the target site
        for job in self.jobHoldingLayer.heldJobs:
            self.assign_condition_matrix[job.id] = {}
            for site in self.siteInfo:
                test_ert = max(int(job.original_ert*float(job.origin_site.gflops/site.gflops)), 1)
                if job.num_required_processors <= site.num_processors and test_ert <= site.max_ert:
                    self.assign_condition_matrix[job.id][site.id] = 1
                else:
                    self.assign_condition_matrix[job.id][site.id] = 0

        #min_wait = float('inf')
        #max_wait = -float('inf')

        for job in self.jobHoldingLayer.heldJobs:
            wait_time_matrix[job.id] = {}
            response_time_matrix[job.id] = {}
            predicted_end_time_matrix[job.id] = {}
            for site in self.siteInfo:
                if self.assign_condition_matrix[job.id][site.id] == 0:
                    continue

                qwait = 0
		# Strategy 2 is INST
		# Strategy 1 is

                if self.strategy != 2:
                    site.queuedJobs.append(job)
                    #print 'To site ', site.id , 'Notifying tentative arrival of job', job.id, 'queued=', len(site.queuedJobs), 'running=', len(site.runningJobs)

                    job.user_estimated_run_time = max(int(job.original_ert*float(job.origin_site.gflops/site.gflops)), 1)
                    job.user_estimated_run_time = (job.user_estimated_run_time/3600 + 1)*3600
                    job.predicted_run_time = job.user_estimated_run_time
                    job.actual_run_time = max(int(job.original_run*float(job.origin_site.gflops/site.gflops)), 1)


                    if len(site.runningJobs) == 0 and len(site.queuedJobs) == 1:
                        qwait = 0
                    else: 
                        site.wait_predictor.notify_arrival_event(job, site.queuedJobs, site.runningJobs)
                        try:
                            qwait = site.wait_predictor.get_prediction(job)
                            if qwait < 0:
			        qwait = 0

                        except Exception:
                            qwait = 0
                            self.qwait_error_count+= 1
                            #print 'Error occured for qwait prediction', self.qwait_error_count
                            pass
                        site.wait_predictor.delete_tentative_arrival_event(job)

                    site.queuedJobs.remove(job)
                response_time = qwait + job.user_estimated_run_time # scaled ERT
                wait_time_matrix[job.id][site.id] = int(qwait)
                response_time_matrix[job.id][site.id] = int(response_time)
                predicted_end_time_matrix[job.id][site.id] = int(response_time) + job.submit_time

                '''
                qwait = 0
                job.user_estimated_run_time = max(int(job.original_ert*float(job.origin_site.gflops/site.gflops)), 1)
                job.user_estimated_run_time = (job.user_estimated_run_time/3600 + 1)*3600
                job.predicted_run_time = job.user_estimated_run_time
                job.actual_run_time = max(int(job.original_run*float(job.origin_site.gflops/site.gflops)), 1)
                

                try:
                    site.queuedJobs.append(job)
                    #print 'To site ', site.id , 'Notifying tentative arrival of job', job.id, 'queued=', len(site.queuedJobs), 'running=', len(site.runningJobs)
                   
                    site.wait_predictor.notify_arrival_event(job, site.queuedJobs, site.runningJobs)
                    qwait = site.wait_predictor.get_prediction(job)
                    site.wait_predictor.delete_tentative_arrival_event(job)
                    site.queuedJobs.remove(job)
                except Exception:
                    self.qwait_error_count+= 1
                    print 'Error occured for qwait prediction', self.qwait_error_count
                    if site.machine.free_processors < job.num_required_processors:
                        qwait = 0
                    pass
                response_time = qwait + job.user_estimated_run_time # scaled ERT
                wait_time_matrix[job.id][site.id] = qwait
                response_time_matrix[job.id][site.id] = response_time
                predicted_end_time_matrix[job.id][site.id] = response_time + job.submit_time
                '''

                site.max_resp_in_cycle = max(site.max_resp_in_cycle, int(response_time))
                min_resp = min(min_resp, response_time_matrix[job.id][site.id])
                max_resp = max(max_resp, response_time_matrix[job.id][site.id])

        for item in rejectJobs:
            wait_time_matrix.pop(item, None)
            response_time_matrix.pop(item, None)
            predicted_end_time_matrix.pop(item, None)

        '''
        * Faster logic for price prediction:
        * For each site, get the maximum predicted response time : site.max_wait_in_cycle
        * Invoke ARIMA on the site for number of slots corrsponding to the max resp time 
        * Use the predictions for all jobs. This will reduce ARIMA models from J*S to S and total time will be dependent only on length of MS-Cycle
        '''
        
        # NL Begin
        if beta != 0.0:
            for site in self.siteInfo:
                if site.max_resp_in_cycle == -float('inf'):
                    continue
                forecast = site.price_predictor.get_batch_forecast(self.current_time,
                                                                   site.max_resp_in_cycle,
                                                                   site.id,
                                                                   )
                if forecast == None:
                    # Time to end simulation, we can't schedule anymore, print statistics and end
                    self.begin_halt = True
                    return None
                else:
                    site.price_forecast = forecast
                    #print "Forecase for site", site.id, site.price_forecast
        # NL End
        for job in self.jobHoldingLayer.heldJobs:
            if job.id in rejectJobs:
                continue
            power_price_matrix[job.id] = {}
            for site in self.siteInfo:
                if self.assign_condition_matrix[job.id][site.id] == 0:
                    continue
                if beta != 0:
                    job.user_estimated_run_time = max(int(job.original_ert*float(job.origin_site.gflops/site.gflops)), 1)
                    job.actual_run_time = max(int(job.original_run*float(job.origin_site.gflops/site.gflops)), 1)

                    job.predicted_run_time = job.user_estimated_run_time
                    #was commented TMPCOMP 
                   
                    price = site.price_predictor.calculate_price_from_batch_forecast(job.get_watts(site.power_per_core),
                                                                                     wait_time_matrix[job.id][site.id],
                                                                                     #job.user_estimated_run_time,
                                                                                     job.actual_run_time,
                                                                                     site.price_forecast)
                    
                    if self.strategy == 3:
                        price = site.price_predictor.get_actual_running_cost(self.current_time+wait_time_matrix[job.id][site.id],
                                                                             self.current_time+wait_time_matrix[job.id][site.id]+job.user_estimated_run_time,
                                                                             job.get_watts(site.power_per_core),
                                                                             square_price=True)
                    else:
                        price = site.price_predictor.get_actual_running_cost(self.current_time+wait_time_matrix[job.id][site.id],
                                                                             self.current_time+wait_time_matrix[job.id][site.id]+job.user_estimated_run_time,
                                                                             job.get_watts(site.power_per_core))
                        #print self.current_time, wait_time_matrix[job.id][site.id], job.user_estimated_run_time, job.get_watts(site.power_per_core), price
                    #print '*** Price of job', job.id, 'on site', site.id, 'with watts=', job.get_watts(site.power_per_core), 'price:', price, 'ERT:', job.user_estimated_run_time

                    power_price_matrix[job.id][site.id] = price

                    #print 'Predicted cost on site', site.id, 'for job', job.id, 'is', '$',float(price/10**6)
                else:
                    power_price_matrix[job.id][site.id] = 1 # Default value
                min_price = min(min_price, power_price_matrix[job.id][site.id])
                max_price = max(max_price, power_price_matrix[job.id][site.id])
        '''
        for job in self.jobHoldingLayer.heldJobs:
            power_price_matrix[job.id] = {}
            for site in self.siteInfo:
                if self.assign_condition_matrix[job.id][site.id] == 0:
                    continue
                print 'Getting price at ', site.id , 'of job', job.id, 
                job.user_estimated_run_time = max(int(job.original_ert*float(job.origin_site.gflops/site.gflops)), 1)
                job.predicted_run_time = job.user_estimated_run_time
                price = site.price_predictor.get_price_prediction(job.submit_time, job.get_watts(site.power_per_core), wait_time_matrix[job.id][site.id], job.user_estimated_run_time, site.id)
                power_price_matrix[job.id][site.id] = int(price)
                print '$',float(price/10**6)
                min_price = min(min_price, power_price_matrix[job.id][site.id])
                max_price = max(max_price, power_price_matrix[job.id][site.id])
        '''
        cost_matrix = {}
        response_normalizer = max(max_resp - min_resp, 1)
        price_normalizer = max(max_price - min_price, 1)

        #TODO: Add ageing factor to prevent starvation
        #TODO: Normalize the costs properly - DONE
        #TODO: Remember to round everything to integers - DONE
        #TODO: Add a quadratic penalty node? - THATS DIFFICULT IN FLOW (FLOW COST IS A SCALAR)



        for job in self.jobHoldingLayer.heldJobs:
            if job.id in rejectJobs:
                continue
            cost_matrix[job.id] = {}
            for site in self.siteInfo:
                if self.assign_condition_matrix[job.id][site.id] == 0:
                    continue
                #print 'Job', job.id, 'on site', site.id, 'response:', response_time_matrix[job.id][site.id], 'wait:', wait_time_matrix[job.id][site.id], 'price:', power_price_matrix[job.id][site.id]
                cost_matrix[job.id][site.id] = alpha*(response_time_matrix[job.id][site.id]-min_resp)/response_normalizer + beta*(power_price_matrix[job.id][site.id]-min_price)/price_normalizer
                cost_matrix[job.id][site.id] = int(cost_matrix[job.id][site.id])
                
        #print '====================='
        for job in self.jobHoldingLayer.heldJobs:
            if job.id in rejectJobs:
                continue
            for site in self.siteInfo:
                if self.assign_condition_matrix[job.id][site.id] == 0:
                    continue
                #if job.origin_site_id == site.id:
                #    print '*',
                #print 'Job', job.id, 'site', site.id, 'response:', response_time_matrix[job.id][site.id], 'wait:', wait_time_matrix[job.id][site.id], 'price:', power_price_matrix[job.id][site.id], 'normalized response:', alpha*(response_time_matrix[job.id][site.id]-min_resp)/response_normalizer, 'normalized price:', beta*(power_price_matrix[job.id][site.id]-min_price)/price_normalizer, 'cost=', cost_matrix[job.id][site.id]
        #print '====================='
        return cost_matrix, wait_time_matrix, response_time_matrix, power_price_matrix, rejectJobs

    def create_job_node_map(self, assignmentCost, jobNodeNumbers):
        jnMap = {}
        current = 0
        for job_id in assignmentCost.keys():
            jnMap[job_id] = jobNodeNumbers[current]
            current += 1
        if current != len(jobNodeNumbers):
            print 'Warning! Mismatch in construct_job_node_map'
        return jnMap

    def create_site_node_map(self, siteNodeNumbers):
        snMap = {}
        current = 0
        for site in self.siteInfo:
            snMap[site.id] = siteNodeNumbers[current]
            current += 1
        if current != len(siteNodeNumbers):
            print 'Warning! Mismatch in construct_site_node_map'
        return snMap

    def create_flow_network(self, assignmentCost):
        G = nx.DiGraph()
        # Add source node edges
        nJobs = len(assignmentCost.keys())
        nSites = len(self.siteInfo)
        
        nodeCount = 1
        jobNodeNumbers = [i+nodeCount for i in range(1,nJobs+1)]
        nodeCount += nJobs
        siteNodeNumbers = [i+nodeCount for i in range(1, nSites+1)]
        nodeCount += nSites

        # need to create a mapping between job ids and node numbers
        # and site ids and site node numbers

        jobMap = self.create_job_node_map(assignmentCost, jobNodeNumbers)
        siteMap = self.create_site_node_map(siteNodeNumbers)
        
        sourceNode = 1
        sinkNode = nodeCount+1
        for jobKey in jobMap.keys():
            G.add_edges_from([(sourceNode, jobMap[jobKey], {'capacity':1, 'weight':0})])
        for jobKey in jobMap.keys():
            for siteKey in siteMap.keys():
                if self.assign_condition_matrix[jobKey][siteKey] == 0:
                    continue
                G.add_edges_from([(jobMap[jobKey], siteMap[siteKey], {'capacity':1, 'weight':assignmentCost[jobKey][siteKey]})])
        for siteKey in siteMap.keys():
            G.add_edges_from([(siteMap[siteKey], sinkNode, {'capacity':self.MaxQ, 'weight':0})])
        return G, sourceNode, sinkNode, jobMap, siteMap

    def copy_job(self, job):
        newjob = Job()
    def post_process_flow(self, flowMap, jobMap, siteMap, rejectJobs, assignCost):
        jobAllocation = []
        for job in jobMap.keys():
            # Find the site (if any) to which it is allocated
            for site in siteMap.keys():
                if self.assign_condition_matrix[job][site] == 0:
                    continue
                if flowMap[jobMap[job]][siteMap[site]] == 1:
                    jobAllocation.append([job, site])
                    #print "allocation cost for job", job, assignCost[job][site]
                    costList = []
                    for s in assignCost[job].keys():
                        costList.append(assignCost[job][s])
                    break
        for job in self.jobHoldingLayer.heldJobs:
            if job.id in rejectJobs:
                jobAllocation.append([job.id, job.origin_site.id])
                print 'Allocating job', job.id, 'to original site:', job.origin_site.id
        return jobAllocation

    '''
    * In the earlier code, each machine handled the start event. But, with a global 
    * event queue this results in repetitive event handlers called one after the 
    * other. So, for metascheduling we need 1 common handler which calls appropriate
    * functions on the site.
    '''
    def handle_start_event(self, event):
        assert isinstance(event, JobStartEvent)
        event.job.sitemap.machine.start_job_handler(event)

    def handle_termination_event(self, event):
        assert isinstance(event, JobTerminationEvent)
        #print 'Term event:', event
        #self.scheduler.cpu_snapshot.printCpuSlices()

        # TODO: Need to change this to invoke termination event of correct site
        # TODO: Also need to handle start event properly
 
        event.job.sitemap.runningJobs.remove(event.job)
        event.job.sitemap.queueSimulator.handle_termination_event(event)
        self.all_terminated_jobs.append(event.job)
        event.job.sitemap.machine.handle_termination_event(event)

        #self.queueSimulator.handle_termination_event(event)
        #queuedJobIDs = [j.id for j in self.event_queue.QueuedJobs]
        #runningJobIDs = [j.id for j in self.event_queue.RunningJobs]

        if EnabledWaitPred:
            event.job.sitemap.wait_predictor.notify_job_termination_event(event.job, event.job.sitemap.queuedJobs, event.job.sitemap.runningJobs)
        '''
        event.job.actual_electricity_price = event.job.sitemap.price_predictor.get_actual_price(event.job.submit_time, 
                                                                                                event.job.get_watts(event.job.target_site.power_per_core),
                                                                                                event.job.start_to_run_at_time-event.job.submit_time, 
                                                                                                event.job.actual_run_time)
        '''

        #print '---> Actual price of job', event.job.id, 'on site', event.job.target_site.id, 'with watts=', event.job.get_watts(event.job.target_site.power_per_core), 'price:', event.job.actual_electricity_price

        if event.job.id > self.historySetSize and event.job.start_to_run_at_time + event.job.actual_run_time < self.target_job_end_time:
             self.total_jobs_for_stats += 1
             event.job.actual_electricity_price = event.job.sitemap.price_predictor.get_actual_running_cost(event.job.start_to_run_at_time,
                                                                                                            event.job.start_to_run_at_time + event.job.actual_run_time,
                                                                                                            event.job.get_watts(event.job.target_site.power_per_core))       
             event.job.actual_electricity_price = float(event.job.actual_electricity_price)
             wait_time = float(event.job.start_to_run_at_time - event.job.submit_time)
             run_time  = float(event.job.actual_run_time)
             self.total_wait_time += wait_time
             self.total_response_time += wait_time + run_time
             self.total_electricity_price += event.job.actual_electricity_price
	     self.total_bounded_slowdown += max(1, (float(wait_time + run_time)/ max(run_time, 10)))
             self.utilization_num += run_time*event.job.num_required_processors
             self.first_job_submit = min(self.first_job_submit, event.job.submit_time)
             self.last_job_finish = max(self.last_job_finish, event.job.start_to_run_at_time + event.job.actual_run_time)

             sid = event.job.target_site.id - 1
             self.site_wait_time[sid] += wait_time
             self.site_response_time[sid] += wait_time + run_time
             self.site_bounded_slowdown[sid] += max(1, (float(wait_time + run_time)/ max(run_time, 10)))
             self.site_electricity_price[sid] += event.job.actual_electricity_price
             self.site_jobs_for_stats[sid] += 1
             self.site_utilization_num[sid] += run_time*event.job.num_required_processors
             self.site_first_job_submit[sid] = min(self.site_first_job_submit[sid], event.job.submit_time)
             self.site_last_job_finish[sid] = max(self.site_last_job_finish[sid], event.job.start_to_run_at_time + event.job.actual_run_time)
             self.GSL.append([event.job.id, event.job.isMeta, event.job.target_site.id, event.job.user_estimated_run_time, event.job.num_required_processors, event.job.user_id, event.job.submit_time, wait_time, run_time, event.job.wait_time_prediction, event.job.actual_electricity_price])

             if event.job.id % 100 == 0:
                 print "============================================================"
                 try:
                     print 'Clean stats:'
                     print 'Average wait time:', float(self.total_wait_time/self.total_jobs_for_stats)/60.0
                     print 'Average response time:', float(self.total_response_time/self.total_jobs_for_stats)/60.0
                     print 'Total electricity price:', float(self.total_electricity_price)
                     print 'Average bounded slowdown:', float(self.total_bounded_slowdown/self.total_jobs_for_stats)
                     print 'System utilization:', 100.0*float(self.utilization_num)/(self.MaxGridProcs*(self.last_job_finish - self.first_job_submit))
                     print 'Total jobs:', self.total_jobs_for_stats
                     print 'Total meta jobs:', self.total_meta_jobs 
                     for sid in range(len(self.siteInfo)):
                         print 'Site', sid
                         print 'Average wait time:', float(self.site_wait_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
                         print 'Average response time:', float(self.site_response_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
                         print 'Total electricity price:', float(self.site_electricity_price[sid])/10**6
                         print 'Average bounded slowdown:', float(self.site_bounded_slowdown[sid]/max(1,self.site_jobs_for_stats[sid]))
                         print 'System utilization:', 100.0*float(self.site_utilization_num[sid])/(self.MaxSiteProcs[sid]*max(1, (self.site_last_job_finish[sid] - self.site_first_job_submit[sid])) )
                         print 'Total jobs:', self.site_jobs_for_stats[sid]
                 except Exception:
                     pass
                 print "============================================================"

        if self.begin_halt:
            #print 'Terminated-dont-check-price:', event.job
            pass


    def handle_prediction_event(self, event):
        assert isinstance(event, JobPredictionIsOverEvent)
        self.queueSimulator.handle_prediction_event(event)

    def modify_job_attributes(self, event, newRequestSize):
        oldRequestSize = event.job.num_required_processors     
        oldERT = event.job.user_estimated_run_time
        newERT = int(oldERT*float(oldRequestSize)/float(newRequestSize))
        oldRunTime = event.job.actual_run_time
        newRunTime = int(oldRunTime*float(oldRequestSize)/float(newRequestSize)) 
        event.job.num_required_processors = newRequestSize
        event.job.user_estimated_run_time = max(newERT, 1)
        event.job.actual_run_time = max(newRunTime, 1)
        event.job.predicted_run_time = max(newERT, 1)    
      
    def run(self):
        while not self.event_queue.is_empty:
            self.event_queue.advance()

    def print_metascheduling_stats(self, target_start_id):
	#rint 'Pickling...'        
	pickle.dump(self.GSL, open(self.pickleFile, "wb"))
        #pickle.dump(self.instLoad, open('instload.pkl', 'wb'))
        #pickle.dump(self.instPower, open('instpower.pkl', 'wb'))
        #pickle.dump(self.instPrice, open('instprice.pkl', 'wb'))
        #pickle.dump(self.instHour, open('insthour.pkl', 'wb'))

        print "============================================================"
        try:
            print 'Overall stats:'
            #print 'Average wait time:', float(self.total_wait_time/self.total_jobs_for_stats)/60.0
            print 'Average response time:', float(self.total_response_time/self.total_jobs_for_stats)/60.0
            print 'Total electricity price:', float(self.total_electricity_price)
            #print 'Average bounded slowdown:', float(self.total_bounded_slowdown/self.total_jobs_for_stats)
            #print 'System utilization:', 100.0*float(self.utilization_num)/(self.MaxGridProcs*(self.last_job_finish - self.first_job_submit))
            #print 'Total jobs:', self.total_jobs_for_stats
            #print 'Total meta jobs:', self.total_meta_jobs 
            for sid in range(len(self.siteInfo)):
                print 'Site', sid
                #print 'Average wait time:', float(self.site_wait_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
                print 'Average response time:', float(self.site_response_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
                print 'Total electricity price:', float(self.site_electricity_price[sid])
                #print 'Average bounded slowdown:', float(self.site_bounded_slowdown[sid]/max(1,self.site_jobs_for_stats[sid]))
                #print 'System utilization:', 100.0*float(self.site_utilization_num[sid])/(self.MaxSiteProcs[sid]*max(1, (self.site_last_job_finish[sid] - self.site_first_job_submit[sid])) )
                #print 'Total jobs:', self.site_jobs_for_stats[sid]
        except Exception:
            pass
        print "============================================================"
        '''
	print 'Pickled!!!'
        print 'Clean stats:'
        print 'Average wait time:', float(self.total_wait_time/self.total_jobs_for_stats)/60.0
        print 'Average response time:', float(self.total_response_time/self.total_jobs_for_stats)/60.0
        print 'Total electricity price:', self.total_electricity_price
	print 'Average bounded slowdown:', float(self.total_bounded_slowdown/self.total_jobs_for_stats)
	print 'System utilization:', 100.0*float(self.utilization_num)/(self.MaxGridProcs*(self.last_job_finish - self.first_job_submit))
        print 'Total jobs:', self.total_jobs_for_stats
        print 'Total meta jobs:', self.total_meta_jobs 
	for sid in range(len(self.siteInfo)):
		print 'Site', sid
		print 'Average wait time:', float(self.site_wait_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
        	print 'Average response time:', float(self.site_response_time[sid]/max(1,self.site_jobs_for_stats[sid]))/60.0
        	print 'Total electricity price:', self.site_electricity_price[sid]
        	print 'Average bounded slowdown:', float(self.site_bounded_slowdown[sid]/max(1,self.site_jobs_for_stats[sid]))
        	print 'System utilization:', 100.0*float(self.site_utilization_num[sid])/(self.MaxSiteProcs[sid]*max(1, (self.site_last_job_finish[sid] - self.site_first_job_submit[sid])) )
        	print 'Total jobs:', self.site_jobs_for_stats[sid]
        '''
        return

        sum_waits = 0.0
        sum_run_times = 0.0
        sum_slowdowns = 0.0
        sum_bounded_slowdowns = 0.0
        sum_power_price = 0.0
        counter = 0.0
        first_grid_job_start = float('inf')
        last_grid_job_finish = 0.0
        total_grid_hours = 0.0
        total_grid_procs = 0.0
        skip_counts = [0]*4
        sum_meta_waits = 0
        print 'Metascheduling Statistics'
        print 'From job:', target_start_id, 'onwards', 'using maxq=', self.MaxQ

        for site in self.siteInfo:
            print 'Site', site.id, 'at', site.location
            site.queueSimulator.scheduler.cpu_snapshot._restore_old_slices()
            req_counters = print_statistics(site.queueSimulator.terminated_jobs, site.queueSimulator.time_of_last_job_submission, target_start_id, site.num_processors)
            total_grid_procs        += site.num_processors
            sum_waits               += req_counters[0]
            sum_run_times           += req_counters[1]
            sum_slowdowns           += req_counters[2]
            sum_bounded_slowdowns   += req_counters[3]
            sum_power_price         += req_counters[4]
            counter                 += req_counters[5]
            total_grid_hours        += req_counters[6]
            first_grid_job_start     = min(first_grid_job_start, req_counters[7])
            last_grid_job_finish     = max(last_grid_job_finish, req_counters[8])
            skip_counts[0]          += req_counters[9]
            skip_counts[1]          += req_counters[10]
            skip_counts[2]          += req_counters[11]
            skip_counts[3]          += req_counters[12]
            sum_meta_waits          += req_counters[13]

        print 'Overall metascheduling stats across all sites:'

        print "Wait (Tw) [minutes]: ", float(sum_waits) / (60 * max(counter, 1))
        print "Meta wait (Tw) [minutes]: ", float(sum_meta_waits) / (60 * max(counter, 1))
        print "Response time (Tw+Tr) [minutes]: ", float(sum_waits + sum_run_times) / (60 * max(counter, 1))
    
        print "Slowdown (Tw+Tr) / Tr: ", sum_slowdowns / max(counter, 1)

        print "Bounded slowdown max(1, (Tw+Tr) / max(10, Tr): ", sum_bounded_slowdowns / max(counter, 1)

        print "Total electricity price ", sum_power_price

        print "Average electricity price", sum_power_price / max(counter, 1)
        
        print "Total utilization of grid", 100.0*total_grid_hours/(total_grid_procs * (last_grid_job_finish-first_grid_job_start))

        print 'Skip counts:', skip_counts

def runMetaScheduler(inputConfiguration, jobs, alpha, beta, algorithm, joblimit,maxq, pfile, mperc, pricepath):
    #print type(jobs)
    metaScheduler = MetaScheduler(inputConfiguration, jobs, alpha, beta, algorithm, joblimit,maxq, pfile, mperc, pricepath)
    metaScheduler.run()
    metaScheduler.print_metascheduling_stats(metaScheduler.historySetSize)
    return metaScheduler

def print_simulator_stats(simulator):
    simulator.scheduler.cpu_snapshot._restore_old_slices()
    # simulator.scheduler.cpu_snapshot.printCpuSlices()
    print_statistics(simulator.terminated_jobs, simulator.time_of_last_job_submission)

# increasing order 
by_finish_time_sort_key   = (
    lambda job : job.finish_time
)

# decreasing order   
#sort by: bounded slow down == max(1, (float(wait_time + run_time)/ max(run_time, 10))) 
by_bounded_slow_down_sort_key = (
    lambda job : -max(1, (float(job.start_to_run_at_time - job.submit_time + job.actual_run_time)/max(job.actual_run_time, 10)))
)

    
def print_statistics(jobs, time_of_last_job_submission, target_start_id, procs):
    assert jobs is not None, "Input file is probably empty."
    
    sum_waits     = 0
    sum_meta_waits = 0
    sum_run_times = 0
    sum_slowdowns           = 0.0
    sum_bounded_slowdowns   = 0.0
    sum_estimated_slowdowns = 0.0
    sum_tail_slowdowns      = 0.0
    sum_power_price         = 0.0
    wait_aae = 0

    counter = tmp_counter = tail_counter = 0
    
    size = len(jobs)
    precent_of_size = int(size / 100)
    
    first_job_submit = float('inf')
    last_job_finish = 0
    total_cpu_hours = 0
    skipped_jobs_id = 0
    skipped_jobs_size1 = 0
    skipped_jobs_size2 = 0
    skipped_jobs_finish_time = 0
    for job in sorted(jobs, key=by_finish_time_sort_key):
        
        if job.id < target_start_id:
            skipped_jobs_id += 1
            continue

        tmp_counter += 1
        if job.user_estimated_run_time == 1 and job.num_required_processors == 1:
            print job
            
        '''
        if job.user_estimated_run_time == 1 and job.num_required_processors == 1: # ignore tiny jobs for the statistics
            skipped_jobs_size1 += 1
            size -= 1
            precent_of_size = int(size / 100)
            continue
        
        if size >= 100 and tmp_counter <= precent_of_size:
            skipped_jobs_size2 += 1
            continue
        '''
        if job.finish_time > time_of_last_job_submission:
            skipped_jobs_finish_time += 1
            break 
        
        
        counter += 1
        first_job_submit = min(first_job_submit, job.submit_time)
        last_job_finish = max(last_job_finish, job.start_to_run_at_time + job.actual_run_time)
        total_cpu_hours += job.actual_run_time*job.num_required_processors
        wait_time = float(job.start_to_run_at_time - job.original_submit_time)
        meta_wait_time = float(job.start_to_run_at_time - job.submit_time)
        run_time  = float(job.actual_run_time)
        estimated_run_time = float(job.user_estimated_run_time)
        wait_aae += abs(job.wait_time_prediction - wait_time)
        
        print 'Job', job.id, 'on site', job.target_site.id, 'wait_time:', wait_time, 'meta_wait_time:', meta_wait_time, 'run_time:', run_time, 'price:', float(job.actual_electricity_price), 'origin site id:', job.origin_site.id

        sum_waits += wait_time
        sum_meta_waits += meta_wait_time
        sum_run_times += run_time
        sum_slowdowns += float(wait_time + run_time) / run_time
        sum_bounded_slowdowns   += max(1, (float(wait_time + run_time)/ max(run_time, 10))) 
        sum_estimated_slowdowns += float(wait_time + run_time) / estimated_run_time
        sum_power_price += float(job.actual_electricity_price)


        if max(1, (float(wait_time + run_time)/ max(run_time, 10))) >= 3:
            tail_counter += 1
            sum_tail_slowdowns += max(1, (float(wait_time + run_time)/ max(run_time, 10)))
            
    sum_percentile_tail_slowdowns = 0.0
    percentile_counter = counter
    
    for job in sorted(jobs, key=by_bounded_slow_down_sort_key):
        wait_time = float(job.start_to_run_at_time - job.submit_time)
        run_time  = float(job.actual_run_time)
        sum_percentile_tail_slowdowns += float(wait_time + run_time) / run_time
        percentile_counter -= 1 # decreamenting the counter 
        if percentile_counter < (0.9 * counter):
            break
        
        
        
    print
    print "STATISTICS: "
    
    print "Wait (Tw) [minutes]: ", float(sum_waits) / (60 * max(counter, 1))
    print "Meta Wait (Tw) [minutes]: ", float(sum_meta_waits) / (60 * max(counter, 1))

    print "Wait AAE:", float(wait_aae)/(60*max(counter, 1))

    print "Response time (Tw+Tr) [minutes]: ", float(sum_waits + sum_run_times) / (60 * max(counter, 1))
    
    print "Slowdown (Tw+Tr) / Tr: ", sum_slowdowns / max(counter, 1)

    print "Bounded slowdown max(1, (Tw+Tr) / max(10, Tr): ", sum_bounded_slowdowns / max(counter, 1)
    
    print "Estimated slowdown (Tw+Tr) / Te: ", sum_estimated_slowdowns / max(counter, 1)


    print "Total electricity price ", sum_power_price
    
    print "Average electricity price", sum_power_price / max(counter, 1)

    print "Tail slowdown (if bounded_sld >= 3): ", sum_tail_slowdowns / max(tail_counter, 1)

    print "   Number of jobs in the tail: ", tail_counter

    print "Tail Percentile (the top 10% sld): ", sum_percentile_tail_slowdowns / max(counter - percentile_counter + 1, 1)    
    
    print "Utilization: ", 100.0*total_cpu_hours/(procs * (last_job_finish - first_job_submit))
    print "Total Number of jobs: ", size
    
    print "Number of jobs used to calculate statistics: ", counter
    print 'Skip counters:', skipped_jobs_id,skipped_jobs_size1,skipped_jobs_size2, skipped_jobs_finish_time 
    print

    req_details = [sum_waits,
                   sum_run_times,
                   sum_slowdowns,
                   sum_bounded_slowdowns,
                   sum_power_price,
                   counter,
                   total_cpu_hours,  
                   first_job_submit, 
                   last_job_finish,
                   skipped_jobs_id,
                   skipped_jobs_size1,
                   skipped_jobs_size2,
                   skipped_jobs_finish_time,
                   sum_meta_waits]
    return req_details
