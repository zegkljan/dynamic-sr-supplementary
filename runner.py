# -*- coding: utf-8 -*-
import argparse
import collections
import gc
import logging
import logging.config
import math
import os
import random
import re
import time
import sys
import resource

import numpy as np
import yaml
from pkg_resources import resource_stream

import evo
import evo.gp
import evo.sr.backpropagation
import evo.sr.bpgp
import evo.utils
import evo.runners.bpgp as bpgp
from evo.runners import text, bounded_integer, bounded_float, float01, \
    PropagateExit


class Runner(bpgp.Runner):
    PARSER_ARG = 'article'

    def __init__(self, subparsers):
        super().__init__(subparsers)

    def _remove_options(self, options):
        for option in options:
            for action in self.parser._actions:
                if vars(action)['option_strings'][0] == option:
                    self.parser._handle_conflict_resolve(None,
                                                         [(option, action)])
                    break

    def setup_input_data_arguments(self):
        super().setup_input_data_arguments()
        self.parser.add_argument(
            '--training-data',
            type=DataSpec,
            required=True,
            metavar='file:x-columns:[y-columns]')

    def setup_output_data_arguments(self):
        super().setup_output_data_arguments()
        self.parser.add_argument(
            '--m-fun',
            help=text(
                'Name of the matlab function the model will be '
                'written to (without extension). Default is '
                '"func_{stage:03}".'),
            type=str,
            default='func_{stage:03}')

    def setup_general_settings_arguments(self):
        super().setup_general_settings_arguments()
        self.parser.add_argument(
            '--stage-time',
            type=bounded_float(0),
            required=True)
        self.parser.add_argument(
            '--stage-iterations',
            type=bounded_integer(0),
            required=True)
        self.parser.add_argument(
            '--reinit-pop',
            action='store_true')
        self.parser.add_argument(
            '--hypermutation-factor',
            type=bounded_float(0),
            default=1)
        self.parser.add_argument(
            '--hypermutation-time',
            type=bounded_float(0),
            default=float('inf'))
        self.parser.add_argument(
            '--hypermutation-iterations',
            type=bounded_integer(0),
            default=float('inf'))

    def load_data(self, params: dict):
        def load(ds: DataSpec, delimiter: str, prefix: str):
            if not os.path.isfile(ds.file):
                logging.error('%s data file %s does not exist or is not a '
                              'file. Exitting.', prefix, ds.file)
                raise PropagateExit(1)
            logging.info('%s data file: %s', prefix, ds.file)
            data = np.loadtxt(ds.file, delimiter=delimiter)
            logging.info('%s data x columns: %s', prefix, ds.x_cols)
            x_data = data[:, ds.x_cols]

            if ds.y_cols is not None:
                logging.info('%s data y columns: %s', prefix, ds.y_cols)
                y_data = data[:, ds.y_cols]
            else:
                y_cols = set(range(data.shape[1]))
                for y in ds.x_cols:
                    y_cols.remove(y)
                y_cols = sorted(list(y_cols))[0:ds.y_cols]
                y_data = data[:, y_cols]

            return x_data, y_data

        dlm = params['delimiter']
        training_ds = params['training-data']

        training_x, training_y = load(training_ds, dlm, 'Training')
        logging.info('Training X data shape (rows, cols): %s', training_x.shape)
        logging.info('Training Y data shape (elements,): %s', training_y.shape)

        return training_x, training_y, None, None

    def prepare_output(self, params: dict):
        output_data = collections.defaultdict(lambda: None)

        output_data['m_fun'] = params['m_fun']

        if params['output_directory'] is None:
            return output_data

        if params['output_directory'] == '-':
            logging.info('No output directory is used.')
            return output_data

        if os.path.isdir(params['output_directory']):
            logging.warning('Output directory %s already exists! Contents '
                            'might be overwritten', params['output_directory'])
        os.makedirs(params['output_directory'], exist_ok=True)
        logging.info('Output directory (relative): %s',
                     os.path.relpath(params['output_directory'], os.getcwd()))
        logging.info('Output directory (absolute): %s',
                     os.path.abspath(params['output_directory']))

        #output_data['y_trn'] = os.path.join(params['output_directory'],
        #                                    'y_trn.txt')
        #output_data['y_tst'] = os.path.join(params['output_directory'],
        #                                    'y_tst.txt')
        output_data['summary'] = os.path.join(params['output_directory'],
                                              'summary.csv')
        output_data['m_func_templ'] = os.path.join(params['output_directory'],
                                                   '{}.m')
        #output_data['stats'] = os.path.join(params['output_directory'],
        #                                    'stats.csv')
        return output_data

    def get_params(self, ns: argparse.Namespace):
        params = dict()

        self.get_logging_params(ns, params)
        self.get_input_params(ns, params)
        self.get_output_params(ns, params)
        self.get_algorithm_params(ns, params)

        return params

    def get_algorithm_params(self, ns, params):
        super().get_algorithm_params(ns, params)
        params['stage_time'] = ns.stage_time
        params['stage_iterations'] = ns.stage_iterations
        params['reinit_pop'] = ns.reinit_pop
        params['hypermutation_factor'] = ns.hypermutation_factor
        params['hypermutation_time'] = ns.hypermutation_time
        params['hypermutation_iterations'] = ns.hypermutation_iterations

    def log_algorithm_params(self, params):
        super().log_algorithm_params(params)
        logging.info('Stage time: %f', params['stage_time'])
        logging.info('Stage iterations: %f', params['stage_iterations'])
        logging.info('Reinit population: %s', params['reinit_pop'])
        logging.info('Hypermutation factor: %f', params['hypermutation_factor'])
        logging.info('Hypermutation time: %f', params['hypermutation_time'])
        logging.info('Hypermutation iterations: %f', params['hypermutation_iterations'])

    def create_fitness(self, params, x, y):
        if params['backpropagation_mode'] not in ['raw', 'none']:
            steps = (params['backpropagation_mode'], params['bprop_steps'])
        else:
            steps = params['bprop_steps']
        fitness = Fitness(
            handled_errors=[],
            train_inputs=x,
            train_output=y[:, 0],
            updater=evo.sr.backpropagation.IRpropMinus(maximize=True),
            steps=steps,
            min_steps=params['bprop_steps_min'],
            fit=True,
            synchronize_lincomb_vars=params['lcf_mode'] == 'synced',
            ## stats=stats,
            fitness_measure=evo.sr.bpgp.ErrorMeasure.R2,
            backpropagate_only=params['lcf_mode'] == 'global'
        )
        fitness.all_ys = y
        return fitness

    def create_algorithm(self, rng, functions, terminals, global_lcs, fitness,
                         population_strategy, reproduction_strategy, callback,
                         stopping_condition, population_initializer,
                         params: dict):
        alg = super().create_algorithm(rng, functions, terminals, global_lcs,
                                       fitness, population_strategy,
                                       reproduction_strategy, callback,
                                       stopping_condition,
                                       population_initializer, params)
        alg.fitness.alg = alg
        alg.stage = 0
        if params['hypermutation_factor'] != 1:
            params2 = {'limits': params['limits'],
                       'pr_x': params['pr_x'],
                       'pr_m': params['pr_m'] * params['hypermutation_factor']}
            rs2 = self.create_reproduction_strategy(
                rng, reproduction_strategy.crossover,
                reproduction_strategy.mutation, functions, terminals, params2)
            alg.callback.hypermutation_reproduction_strategy = rs2
            alg.callback.hypermutation_time = params['hypermutation_time']
            alg.callback.hypermutation_iterations = params['hypermutation_iterations']
        return alg

    def create_stopping_condition(self, params):
        if math.isinf(params['time']) and math.isinf(params['generations']):
            logging.warning('Both time and generational stopping condition '
                            'will never be met. Algorithm must be terminated '
                            'externally.')
        time_stop = evo.gp.Gp.time(params['time'])
        generations_stop = evo.gp.Gp.generations(params['generations'])

        def stage_stop(a):
            if a.stage >= a.fitness.all_ys.shape[1]:
                    raise evo.StopEvolution('Final stage completed.')

        if params['generation_time_combinator'] == 'any':
            stop = evo.gp.Gp.any(time_stop, generations_stop, stage_stop)
        elif params['generation_time_combinator'] == 'all':
            stop = evo.gp.Gp.all(time_stop, generations_stop, stage_stop)
        else:
            raise ValueError('Invalid generation-time-combinator')

        return stop

    def create_callback(self, params):
        class Cb(evo.gp.Callback):
            def __init__(self):
                self.orig_reproduction_strategy = None
                self.hypermutation_reproduction_strategy = None
                self.hypermutation_time = None
                self.hypermutation_iterations = None
                self.hypermutation_start_iteration = -float('inf')

            def start(self, alg):
                alg.stage = 0
                self.orig_reproduction_strategy = alg.reproduction_strategy

            def check_stage(self, alg):
                if (alg.get_runtime() >= (alg.stage + 1) * params['stage_time'] or
                    alg.iterations >= (alg.stage + 1) * params['stage_iterations']):
                    logging.info('Stage %d finished.', alg.stage)
                    stages = alg.fitness.all_ys.shape[1]
                    alg.stage += 1
                    if alg.stage >= stages:
                        raise evo.StopEvolution('Final stage completed.')
                    y = alg.fitness.all_ys[:, alg.stage]
                    alg.fitness.train_output = y
                    alg.fitness.ssw = np.sum((y - y.mean()) ** 2)
                    alg.fitness.bsf = None
                    if params['reinit_pop']:
                        del alg.population
                        alg.population = alg.population_initializer.initialize(
                            alg.pop_strategy.get_parents_number(), alg.limits)
                    if self.hypermutation_reproduction_strategy is not None:
                        logging.info('Starting hypermutation.')
                        self.hypermutation_start_iteration = alg.iterations
                        alg.reproduction_strategy = self.hypermutation_reproduction_strategy

            def iteration_start(self, alg):
                if not params['reinit_pop']:
                    self.check_stage(alg)
                if (self.hypermutation_reproduction_strategy is not None and
                        alg.reproduction_strategy is self.hypermutation_reproduction_strategy and
                        (alg.get_runtime() >= alg.stage * params['stage_time'] + self.hypermutation_time or
                         alg.iterations > self.hypermutation_start_iteration + self.hypermutation_iterations)):
                    logging.info('Stopping hypermutation.')
                    alg.reproduction_strategy = self.orig_reproduction_strategy

            def before_eval_individual(self, alg, _):
                if not params['reinit_pop']:
                    self.check_stage(alg)
                if (self.hypermutation_reproduction_strategy is not None and
                        alg.reproduction_strategy is self.hypermutation_reproduction_strategy and
                        (alg.get_runtime() >= alg.stage * params['stage_time'] + self.hypermutation_time or
                         alg.iterations > self.hypermutation_start_iteration + self.hypermutation_iterations)):
                    logging.info('Stopping hypermutation.')
                    alg.reproduction_strategy = self.orig_reproduction_strategy

            def iteration_end(self, alg):
                self.check_stage(alg)
                if (self.hypermutation_reproduction_strategy is not None and
                        alg.reproduction_strategy is self.hypermutation_reproduction_strategy and
                        (alg.get_runtime() >= alg.stage * params['stage_time'] + self.hypermutation_time or
                         alg.iterations > self.hypermutation_start_iteration + self.hypermutation_iterations)):
                    logging.info('Stopping hypermutation.')
                    alg.reproduction_strategy = self.orig_reproduction_strategy
                logging.info('Memory usage: %d MB / %d MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, resource.getrlimit(resource.RLIMIT_AS)[0] // (1024 * 1024))
                if alg.iterations % 1 == 0:
                    t1 = time.time()
                    objs = gc.collect()
                    t2 = time.time()
                    logging.info('Collecting garbage... %d objects collected in %f s', objs, t2 - t1)

        if params['backpropagation_mode'] != 'none':
            class Cb2(Cb):

                def iteration_start(self, alg):
                    super().iteration_start(alg)
                    for i in alg.population:
                        i.set_fitness(None)

            cb = Cb2()
        else:
            cb = Cb()

        return cb

    def postprocess(self, algorithm, x_data_trn, y_data_trn, x_data_tst,
                    y_data_tst, output, ns: argparse.Namespace):
        bsfs = algorithm.fitness.bsfs
        del algorithm

        if output['summary'] is not None:
            with open(output['summary'], 'w') as out:
                print('r2,mse,mae,wcae,nodes,depth,time,iteration,'
                      'fitness_eval,stage,model', file=out)
            for i in range(len(bsfs)):
                logging.info('Postprocessing bsf No. %d/%d...', i + 1, i + len(bsfs))
                self.postprocess_single(
                    bsfs[0], x_data_trn, y_data_trn, output,
                    len(bsfs) == 1 or
                    bsfs[0].data['stage'] != bsfs[1].data['stage'])
                del bsfs[0]
                logging.info('Memory usage: %d MB / %d MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, resource.getrlimit(resource.RLIMIT_AS)[0] // (1024 * 1024))
                t1 = time.time()
                objs = gc.collect()
                t2 = time.time()
                logging.info('Collecting garbage... %d objects collected in %f s', objs, t2 - t1)

    def postprocess_single(self, bsf, x_data_trn, y_data_trn, output,
                           last_in_stage):
        try:
            y = self.eval_individual(x_data_trn, bsf.bsf)
        except Exception as e:
            logging.exception('Postprocessing failed.')
            return
        stage = bsf.data['stage']
        r2 = bpgp.r2(y_data_trn[:, stage], y)
        mse = bpgp.mse(y_data_trn[:, stage], y)
        mae = bpgp.mae(y_data_trn[:, stage], y)
        wcae = bpgp.wcae(y_data_trn[:, stage], y)
        nodes = sum(g.get_subtree_size() for g in bsf.bsf.genotype)
        depth = max(g.get_subtree_depth() for g in bsf.bsf.genotype)
        time = bsf.data['runtime']
        iteration = bsf.iteration
        fitness_eval = bsf.eval_count

        model_str = bsf.bsf.serialize(element_delimiter='|')
        #model_str = evo.sr.bpgp.full_model_str(bsf.bsf,
        #                                       num_format='repr',
        #                                       newline_genes=False)
        with open(output['summary'], 'a') as summary_file:
            print('{r2},{mse},{mae},{wcae},{nodes},{depth},{time},{iteration},'
                  '{fitness_eval},{stage},"{model}"'.format(
                      r2=r2,
                      mse=mse,
                      mae=mae,
                      wcae=wcae,
                      nodes=nodes,
                      depth=depth,
                      time=time,
                      iteration=iteration,
                      fitness_eval=fitness_eval,
                      stage=stage,
                      model=model_str),
                  file=summary_file)
        #if output['y_trn'] is not None:
        #    np.savetxt(output['y_trn'], y_trn, delimiter=',')
        if output['m_func_templ'] is not None:
            m_fun_name = output['m_fun'].format(stage=stage,
                                                iteration=iteration)
            m_fun_fn = output['m_func_templ'].format(m_fun_name)
            logging.info('Writing matlab function to %s', m_fun_fn)
            with open(m_fun_fn, 'w') as out:
                print(bsf.bsf.to_matlab(m_fun_name), file=out)

    def eval_individual(self, x, individual):
        for g in individual.genotype:
            g.clear_cache()

        if individual.genes_num == 1:
            outputs = individual.genotype[0].eval(args=x)
        else:
            outputs = [g.eval(args=x) for g
                       in individual.genotype]
            outputs = evo.utils.column_stack(*outputs)

        intercept = individual.intercept
        coefficients = individual.coefficients
        if coefficients is not None:
            if outputs.ndim == 1 or outputs.ndim == 0:
                outputs = outputs * coefficients
            else:
                outputs = outputs.dot(coefficients)
        if intercept is not None:
            outputs = outputs + intercept

        if outputs.size == 1:
            outputs = np.repeat(outputs, x.shape[0])
        return outputs


class DataSpec(object):
    def __init__(self, spec: str):
        self.file = None
        self.x_cols = None
        self.y_cols = None

        parts = spec.split(':')
        if len(parts) == 3:
            self.file = parts[0]
            try:
                self.x_cols = list(map(int, parts[1].split(',')))
            except ValueError:
                raise argparse.ArgumentTypeError('x-columns spec is invalid')
            if parts[2] == '':
                return
            try:
                self.y_cols = list(map(int, parts[2].split(',')))
            except ValueError:
                raise argparse.ArgumentTypeError('y-column spec is invalid')
        else:
            raise argparse.ArgumentTypeError('Data specification contains '
                                             'invalid number of parts '
                                             '(separated by colon). The number '
                                             'of parts must be 3 (two '
                                             'colons, file name, x-columns '
                                             'spec and y-columns spec).')

    def __repr__(self):
        if self.x_cols is None and self.y_col is None:
            return '{}(\'{}\')'.format(self.__class__.__name__, self.file)
        return '{}(\'{}:{}:{}\')'.format(self.__class__.__name__, self.file,
                                         self.x_cols, self.y_col)


class Fitness(evo.sr.bpgp.RegressionFitness):
    def compute_individual_data(self, individual):
        for g in individual.genotype:
            g.clear_cache()
            def fn(n):
                if 'd_bias' in n.data:
                    del n.data['d_bias']
                if 'd_weights' in n.data:
                    del n.data['d_weights']
                if getattr(n, 'argument', None) is not None:
                    n.argument = None
            g.preorder(fn)
        return {
            'runtime': self.alg.get_runtime(),
            'stage': self.alg.stage,
            'y': self.train_output
        }


class RootParser(object):
    def __init__(self):
        import evo.__main__
        parser = argparse.ArgumentParser(
            prog='evo',
            formatter_class=evo.__main__.
                PreserveWhiteSpaceWrapRawTextHelpFormatter
        )
        # common options
        parser.add_argument('--version',
                            help=text('Print version and exit.'),
                            action='version',
                            version=evo.__version__)
        parser.add_argument('--logconf',
                            help=text('logging configuration file (yaml '
                                      'format)'),
                            default=None)

        self.parser = parser

        # subcommands
        subparsers = parser.add_subparsers(title='algorithms',
                                           metavar='<algorithm>',
                                           dest='algorithm')
        self.parser_handlers = {p: h for p, h in [
            (lambda r: (r.PARSER_ARG, r))(Runner(subparsers))
        ]}

    def parse(self):
        return self.parser.parse_args()

    def handle(self, ns: argparse.Namespace):
        return self.parser_handlers[ns.algorithm].handle(ns)


def main():
    print('Arguments: {}'.format(sys.argv[1:]), file=sys.stderr)
    memlimit_soft, memlimit_hard = resource.getrlimit(resource.RLIMIT_AS)
    print('resource soft: {}'.format(memlimit_soft), file=sys.stderr)
    print('resource hard: {}'.format(memlimit_hard), file=sys.stderr)
    if 'PBS_RESC_TOTAL_MEM' in os.environ:
        pbs_memlimit = int(os.environ['PBS_RESC_TOTAL_MEM'])
        print('PBS_RESC_TOTAL_MEM: {}'.format(pbs_memlimit), file=sys.stderr)
        resource.setrlimit(resource.RLIMIT_AS, (pbs_memlimit - 50 * 1024 * 1024,
                                                memlimit_hard))
        memlimit_soft, memlimit_hard = resource.getrlimit(resource.RLIMIT_AS)
        print('new resource soft: {}'.format(memlimit_soft), file=sys.stderr)
    parser = RootParser()
    ns = parser.parse()
    status = parser.handle(ns)
    if status is not None:
        return status


if __name__ == '__main__':
    sys.exit(main())
