#! /usr/bin/env python2.4

import sys
if __debug__:
    import warnings
    #warnings.warn("Running in debug mode, this will be slow... try 'python2.4 -O %s'" % sys.argv[0])

from base.workload_parser import parse_lines
from base.prototype import _job_inputs_to_jobs
from base.prototype import Site, SiteData
from schedulers.simulator import run_simulator
from schedulers.metaScheduler import runMetaScheduler

import optparse

from schedulers.fcfs_scheduler import FcfsScheduler

from schedulers.conservative_scheduler import ConservativeScheduler
from schedulers.double_conservative_scheduler import DoubleConservativeScheduler

from schedulers.easy_scheduler import EasyBackfillScheduler

from schedulers.double_easy_scheduler import DoubleEasyBackfillScheduler
from schedulers.head_double_easy_scheduler import HeadDoubleEasyScheduler
from schedulers.tail_double_easy_scheduler import TailDoubleEasyScheduler

from schedulers.greedy_easy_scheduler import GreedyEasyBackfillScheduler

from schedulers.easy_plus_plus_scheduler import EasyPlusPlusScheduler
from schedulers.common_dist_easy_plus_plus_scheduler import CommonDistEasyPlusPlusScheduler
from schedulers.alpha_easy_scheduler import AlphaEasyScheduler

from schedulers.shrinking_easy_scheduler import ShrinkingEasyScheduler


from schedulers.easy_sjbf_scheduler import EasySJBFScheduler
from schedulers.reverse_easy_scheduler import ReverseEasyScheduler
from schedulers.perfect_easy_scheduler import PerfectEasyBackfillScheduler
from schedulers.double_perfect_easy_scheduler import DoublePerfectEasyBackfillScheduler

from schedulers.lookahead_easy_scheduler import LookAheadEasyBackFillScheduler

from schedulers.orig_probabilistic_easy_scheduler import OrigProbabilisticEasyScheduler



def parse_options():
    parser = optparse.OptionParser()
    #parser.add_option("--num-processors", type="int", help="the number of available processors in the simulated parallel machine")
    parser.add_option("--input-file", \
                      help="a file in the standard workload format: http://www.cs.huji.ac.il/labs/parallel/workload/swf.html, if '-' read from stdin")
    parser.add_option("--scheduler", 
                      help="1) FcfsScheduler, 2) ConservativeScheduler, 3) DoubleConservativeScheduler, 4) EasyBackfillScheduler, 5) DoubleEasyBackfillScheduler, 6) GreedyEasyBackfillScheduler, 7) EasyPlusPlusScheduler, 8) ShrinkingEasyScheduler, 9) LookAheadEasyBackFillScheduler,  10) EasySJBFScheduler, 11) HeadDoubleEasyScheduler, 12) TailDoubleEasyScheduler, 13) OrigProbabilisticEasyScheduler, 14) ReverseEasyScheduler,  15) PerfectEasyBackfillScheduler, 16)DoublePerfectEasyBackfillScheduler, 17) ProbabilisticNodesEasyScheduler, 18) AlphaEasyScheduler, 19)DoubleAlphaEasyScheduler 20)ProbabilisticAlphaEasyScheduler")
    
    parser.add_option("--alpha", type="float")
    parser.add_option("--beta", type="float")
    parser.add_option("--algorithm", type="int", help="1) MCMF 2) baseline random")
    parser.add_option("--joblimit", type="int")
    parser.add_option("--maxq", type="int")    
    parser.add_option("--pfile")
    parser.add_option("--mperc", type="float")
    parser.add_option("--pricepath", type="string")

    options, args = parser.parse_args()

    if options.input_file is None:
        parser.error("missing input file")

    if options.scheduler is None:
         parser.error("missing scheduler")

    if args:
        parser.error("unknown extra arguments: %s" % args)

    return options

def main():
    options = parse_options()

    if options.input_file == "-":
        input_file = sys.stdin
    else:
        input_file = open(options.input_file)
    try:
        print "...." 
     
        # Site 1
        Blacklight = SiteData()
        Blacklight.location = "Pittsburgh"
        Blacklight.name = "Blacklight"
        Blacklight.cores = 4096
        Blacklight.powerPerCore = 87.89 # Watts
        Blacklight.timeZoneLag = 0
        Blacklight.gflops = 9.0
        Blacklight.max_ert = 86400
        Blacklight.max_req = 1024

        # Site 2
        Darter = SiteData()
        Darter.location = "Tennessee"
        Darter.name = "Darter"
        Darter.cores = 11968
        Darter.powerPerCore = 30.588624 
        Darter.timeZoneLag = -1
        Darter.gflops = 20.79
        Darter.max_ert = 86400
        Darter.max_req = 2048

        # Site 3
        Gordon = SiteData()
        Gordon.location = "San Diego"
        Gordon.name = "Gordon"
        Gordon.cores = 16160
        Gordon.powerPerCore = 22.178218
        Gordon.timeZoneLag = -3
        Gordon.gflops = 21.10
        Gordon.max_ert = 172800
        Gordon.max_req = 1024

        # Site 4
        Trestles = SiteData()
        Trestles.location = "San Diego"
        Trestles.name = "Trestles"
        Trestles.cores = 10368
        Trestles.powerPerCore = 42.6697
        Trestles.timeZoneLag = -3
        Trestles.gflops = 9.64
        Trestles.max_ert = 172800
        Trestles.max_req = 1024

        # Site 5
        Mason = SiteData()
        Mason.location = "Indiana"
        Mason.name = "Mason"
        Mason.cores = 576
        Mason.powerPerCore = 39.952891
        Mason.timeZoneLag = 0
        Mason.gflops = 7.43
        Mason.max_ert = 172800
        Mason.max_req = 128

        # Site 6
        Lonestar = SiteData()
        Lonestar.location = "Texas"
        Lonestar.name = "Lonestar"
        Lonestar.cores = 22656
        Lonestar.powerPerCore = 15.83
        Lonestar.timeZoneLag = 2
        Lonestar.gflops = 13.33
        Lonestar.max_ert = 86400
        Lonestar.max_req = 4096

        # Site 7
        Queenbee = SiteData()
        Queenbee.location = "Louisiana"
        Queenbee.name = "Queenbee"
        Queenbee.cores = 5440
        Queenbee.powerPerCore = 16.25
        Queenbee.timeZoneLag = -1
        Queenbee.gflops = 9.38
        Queenbee.max_ert = 259200
        Queenbee.max_req = 2048

        # Site 8
        Steele = SiteData()
        Steele.location = "Indiana"
        Steele.name = "Steele"
        Steele.cores = 4992
        Steele.powerPerCore = 83.75
        Steele.timeZoneLag = 0
        Steele.gflops = 13.34
        Steele.max_ert = 259200
        Steele.max_req = 1024

        SiteList = [Blacklight, 
                    Darter, 
                    Gordon,
                    Trestles, 
                    Mason, 
                    Lonestar, 
                    Queenbee, 
                    Steele]
        input = []
        siteCount = 1
        maxProcessors = 0
        sumProcessors = 0
        for site in SiteList:
            maxProcessors = max(maxProcessors, site.cores)
            sumProcessors = maxProcessors + site.cores
            input.append(Site(siteCount, 
                              site.location, 
                              site.name,
                              site.timeZoneLag, 
                              site.cores, 
                              EasyBackfillScheduler(site.cores), 
                              site.powerPerCore,
                              site.gflops,
                              site.max_ert,
                              site.max_req))
            siteCount += 1
        
        alpha = options.alpha
        beta = options.beta
        algorithm = options.algorithm
        joblimit = options.joblimit
        ''' 
        * Algorithm 1 - MCMF
        * Algorithm 2 - INST
	* 3 - two price
        '''

        runMetaScheduler(input, jobs = _job_inputs_to_jobs(parse_lines(input_file), maxProcessors), alpha=alpha, beta=beta, algorithm=algorithm, joblimit=options.joblimit,maxq=options.maxq, pfile= options.pfile, mperc=options.mperc, pricepath=options.pricepath)


        #print "Num of Processors: ", sumProcessors
        print "Alpha=", alpha, "Beta=", beta
        print "Input file: ", options.input_file
        
    finally:
        if input_file is not sys.stdin:
            input_file.close()


if __name__ == "__main__":# and not "time" in sys.modules:
    try:
        import psyco
        psyco.full()
    except ImportError:
        print "Psyco not available, will run slower (http://psyco.sourceforge.net)"
    main()
