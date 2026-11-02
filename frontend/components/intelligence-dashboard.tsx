'use client'

import { useMemo, useState, useEffect, type ReactNode } from 'react'
import useSWR from 'swr'
import { AnimatePresence, motion } from 'motion/react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  ChartNoAxesCombined,
  Database,
  Images,
  Network,
  RefreshCw,
  Search,
  ShieldCheck,
  Target,
  Zap,
} from 'lucide-react'
import { fetchBackend, numberValue, records, type ApiRecord } from '@/lib/backend-api'

const cyan = '#596b45'
const violet = '#b58b32'
const coral = '#a95543'
const amber = '#c69b45'
const emerald = '#71834f'

const preprocessingAlgorithms = [
  { id: 'gcn', name: 'Graph Convolutional Network (GCN)', chartName: 'GCN' },
  { id: 'graphsage', name: 'GraphSAGE', chartName: 'GraphSAGE' },
  { id: 'gru', name: 'Temporal GRU', chartName: 'GRU' },
  { id: 'logistic', name: 'Logistic Regression', chartName: 'Logistic' },
  { id: 'risk_scorer', name: 'Calibrated Risk Scorer (Platt-Calibrated GRU)', chartName: 'Risk Scorer' },
]

const preprocessingMetrics = [
  { id: 'rmse', label: 'RMSE', higherIsBetter: false },
  { id: 'mse', label: 'MSE', higherIsBetter: false },
  { id: 'mae', label: 'MAE', higherIsBetter: false },
  { id: 'mape', label: 'MAPE', higherIsBetter: false },
  { id: 'r2', label: 'R²', higherIsBetter: true },
  { id: 'nrmse', label: 'NRMSE', higherIsBetter: false },
  { id: 'medae', label: 'MedAE', higherIsBetter: false },
  { id: 'bias', label: 'Bias / ME', higherIsBetter: false, closerToZero: true },
]

const preprocessingErrorCharts = [
  { id: 'rmse', title: 'Root Mean Square Error (RMSE)', axisLabel: 'RMSE' },
  { id: 'mse', title: 'Mean Squared Error (MSE)', axisLabel: 'MSE' },
  { id: 'mae', title: 'Mean Absolute Error (MAE)', axisLabel: 'MAE' },
  { id: 'r2', title: 'Coefficient of Determination (R²)', axisLabel: 'R²' },
  { id: 'nrmse', title: 'Normalized Root Mean Square Error (NRMSE)', axisLabel: 'NRMSE' },
  { id: 'medae', title: 'Median Absolute Error (MedAE)', axisLabel: 'MedAE' },
  { id: 'bias', title: 'Bias / Mean Error', axisLabel: 'Signed mean error' },
]

function savedRegressionMetric(data: ApiRecord | undefined, metricId: string, modelId: string): ApiRecord | undefined {
  const groups = (data?.groups || {}) as Record<string, Array<ApiRecord>>
  const regression = groups.regression || []
  const row = regression.find((item) => String(item.metric) === metricId)
  const results = (row?.results || {}) as Record<string, ApiRecord>
  return results[modelId]
}

function comparisonMetricValue(result: ApiRecord | undefined): number | null {
  if (String(result?.status || '') !== 'calculated') return null
  return numberValue(result, ['value'])
}

type ViewId = 'overview' | 'explorer' | 'models' | 'inference' | 'metrics' | 'graphs'
type Feed = { mutate: () => void }

const nav: Array<{ id: ViewId; label: string; icon: typeof Activity; index: string }> = [
  { id: 'overview', label: 'Overview', icon: Activity, index: '01' },
  { id: 'explorer', label: 'Explorer', icon: Target, index: '02' },
  { id: 'models', label: 'Model Lab', icon: BrainCircuit, index: '03' },
  { id: 'inference', label: 'Inference', icon: Zap, index: '04' },
  { id: 'metrics', label: 'Metrics', icon: ChartNoAxesCombined, index: '05' },
  { id: 'graphs', label: 'Graph Gallery', icon: Images, index: '06' },
]

function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`glass-card ${className}`}>{children}</section>
}

function Label({ children }: { children: ReactNode }) {
  return <span className="eyebrow">{children}</span>
}

function pick(row: ApiRecord | undefined, keys: string[]) {
  if (!row) return undefined
  const key = keys.find((candidate) => row[candidate] !== undefined)
  return key ? row[key] : undefined
}

function text(row: ApiRecord | undefined, keys: string[], fallback = '—') {
  if (!row) return fallback
  const value = pick(row, keys)
  return value === undefined || value === null || value === '' ? fallback : String(value)
}

function num(row: ApiRecord | undefined, keys: string[], decimals = 4, fallback = '—'): string {
  if (!row) return fallback
  const val = numberValue(row, keys)
  if (val === null || val === undefined) return fallback
  return val.toFixed(decimals)
}

function formatInt(value: unknown, fallback = '—'): string {
  if (value === undefined || value === null) return fallback
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed.toLocaleString() : fallback
}

function Empty({ message = 'No records were returned by this endpoint.' }: { message?: string }) {
  return (
    <div className="empty-state">
      <Database size={24} />
      <span>{message}</span>
    </div>
  )
}

function Loading() {
  return (
    <div className="loading-grid">
      <div />
      <div />
      <div />
    </div>
  )
}

function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  return (
    <div className="error-state">
      <AlertTriangle size={18} />
      <span>{error.message}</span>
      <button className="sync-btn" onClick={retry}>
        Retry
      </button>
    </div>
  )
}

function SectionHead({ eyebrow, title, meta }: { eyebrow: string; title: string; meta?: string }) {
  return (
    <div className="section-head">
      <div>
        <Label>{eyebrow}</Label>
        <h2>{title}</h2>
      </div>
      {meta && <span className="stage-live">{meta}</span>}
    </div>
  )
}

/* Graph views are built only from the saved graph summary and edge artifacts. */
function GraphSummaryPanel({ overviewData }: { overviewData?: ApiRecord }) {
  const topology = (overviewData?.graph_topology as ApiRecord) || undefined
  return (
    <div className="flow-field graph-summary-panel">
      <div className="flow-field-head">
        <div className="flow-head-left"><Network size={16} /> Graph summary from saved artifacts</div>
        <span className="flow-head-right">FULL-GRAPH DESCRIPTIVE</span>
      </div>
      <div className="graph-summary-grid">
        <div><span>Directed edges</span><strong>{formatInt(overviewData?.relationships)}</strong></div>
        <div><span>Weak components</span><strong>{formatInt(pick(topology, ['weak_components']))}</strong></div>
        <div><span>Louvain communities</span><strong>{formatInt(pick(topology, ['louvain_communities']))}</strong></div>
        <div><span>Modularity</span><strong>{num(topology, ['louvain_modularity'], 4)}</strong></div>
      </div>
      <p>These are full-graph descriptive statistics. They are not historical, as-of-timestep features and do not identify wallet owners.</p>
    </div>
  )
}

/* =========================================================================
   Distinctive Asymmetric Hero Section (100% Dynamic Metadata, Zero Hardcoding)
   ========================================================================= */
function Hero({
  overviewData,
}: {
  overviewData?: ApiRecord
}) {
  const datasetOverview = (overviewData?.dataset_overview as ApiRecord) || undefined
  const topology = (overviewData?.graph_topology as ApiRecord) || undefined

  const evalSplit = datasetOverview?.test_timesteps ? `Timesteps ${datasetOverview.test_timesteps}` : '—'
  const supervisedCount = datasetOverview?.test_labeled_count
    ? `${Number(datasetOverview.test_labeled_count).toLocaleString()} Labeled`
    : '—'
  const baseIllicit = overviewData?.illicit_rate
    ? `${overviewData.illicit_rate} Base Rate`
    : datasetOverview?.test_illicit_prevalence_pct
    ? `${Number(datasetOverview.test_illicit_prevalence_pct).toFixed(2)}% Base Rate`
    : '—'
  const communities = topology?.louvain_communities
    ? `${topology.louvain_communities} Communities`
    : 'Graph Verified'
  const relCount = overviewData?.relationships ? Number(overviewData.relationships).toLocaleString() : 'observed'

  return (
    <section className="forensic-hero">
      <div className="hero-col-left">
        <div className="hero-eyebrow">
          <span className="live-dot" />
          <span>BLOCKCHAIN FORENSICS · ELLIPTIC++ BENCHMARK</span>
        </div>

        <h1 className="hero-heading">
          Trace the flow<span className="headline-period">.</span>
        </h1>

        <p className="hero-description">
          Follow continuous evidentiary signals across {relCount} directed Bitcoin transaction edges in held-out timesteps, evaluating graph neural network embeddings against supervised benchmark labels.
        </p>

        <div className="hero-meta-grid">
          <div className="meta-tile">
            <span className="meta-kicker">EVALUATION SPLIT</span>
            <strong>{evalSplit}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">SUPERVISED SUPPORT</span>
            <strong>{supervisedCount}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">BASE ILLICIT PREVALENCE</span>
            <strong className="text-coral">{baseIllicit}</strong>
          </div>
          <div className="meta-tile">
            <span className="meta-kicker">TOPOLOGY PARTITIONS</span>
            <strong className="text-cyan">{communities}</strong>
          </div>
        </div>
      </div>

      <div className="hero-col-right">
      <GraphSummaryPanel overviewData={overviewData} />
      </div>
    </section>
  )
}

/* =========================================================================
   View 01: Overview (100% Dynamic KPI Blocks + Clear Model-Insights Section)
   ========================================================================= */
function Overview({
  overviewData,
  modelsData,
  suspiciousData,
  loading,
  error,
  retry,
  onNavigate,
}: {
  overviewData?: ApiRecord
  modelsData?: ApiRecord | ApiRecord[]
  suspiciousData?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
  onNavigate: (view: ViewId) => void
}) {
  const row = overviewData
  const datasetOverview = (row?.dataset_overview as ApiRecord) || undefined
  const suspiciousRecords = records(suspiciousData)

  // Derive Champion Model dynamically from /api/models
  const modelList = Array.isArray((modelsData as any)?.models)
    ? (modelsData as any).models
    : records(modelsData)
  const championModel = modelList.find((m: any) => m.id === 'risk_scorer') || modelList[0]

  const champPrAuc = championModel
    ? num(championModel, ['pr_auc', 'benchmark_pr_auc', 'auc'], 4)
    : '—'
  const champRecall = championModel?.recall !== undefined
    ? `${(Number(championModel.recall) * 100).toFixed(2)}%`
    : '—'
  const champPrecision = championModel?.precision !== undefined
    ? `${(Number(championModel.precision) * 100).toFixed(2)}%`
    : '—'
  const champAccuracy = championModel?.accuracy !== undefined
    ? `${(Number(championModel.accuracy) * 100).toFixed(2)}%`
    : '—'

  return (
    <div className="view-overview">
      <Hero overviewData={row} />

      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorState error={error} retry={retry} />
      ) : (
        <>
          {/* Four Readable Dynamic KPI Blocks */}
          <div className="kpi-quartet">
            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
              <i className="kpi-accent" style={{ background: cyan }} />
              <span className="kpi-label">Observed Transactions</span>
              <strong className="kpi-value">{formatInt(row?.total_transactions || row?.transactions)}</strong>
              <span className="kpi-hint">Full dataset transaction nodes</span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <i className="kpi-accent" style={{ background: violet }} />
              <span className="kpi-label">Directed Relationships</span>
              <strong className="kpi-value">{formatInt(row?.relationships || row?.edges)}</strong>
              <span className="kpi-hint">Directed transaction flow edges</span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
              <i className="kpi-accent" style={{ background: emerald }} />
              <span className="kpi-label">Held-Out Test Set</span>
              <strong className="kpi-value">{formatInt(row?.test_set || datasetOverview?.held_out_test_transactions)}</strong>
              <span className="kpi-hint">
                {datasetOverview?.test_timesteps ? `Transactions in Timesteps ${datasetOverview.test_timesteps}` : 'Held-out test partition'}
              </span>
            </motion.div>

            <motion.div className="kpi-card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <i className="kpi-accent" style={{ background: coral }} />
              <span className="kpi-label">Base Illicit Rate</span>
              <strong className="kpi-value text-coral">{text(row, ['illicit_rate', 'suspicious_rate'])}</strong>
              <span className="kpi-hint">
                {datasetOverview?.test_illicit_count && datasetOverview?.test_labeled_count
                  ? `${formatInt(datasetOverview.test_illicit_count)} illicit in ${formatInt(datasetOverview.test_labeled_count)} labeled test`
                  : 'Empirical test prevalence'}
              </span>
            </motion.div>
          </div>

          {/* ONE Clear Model-Insights Section (Zero demo data, derived purely from API) */}
          <Card className="overview-summary-card">
            <SectionHead
              eyebrow="Empirical Detection Benchmark &amp; Risk Composition"
              title="Model Intelligence &amp; Detection Insights"
              meta={datasetOverview?.test_timesteps ? `TIMESTEPS ${datasetOverview.test_timesteps} BENCHMARK` : 'BENCHMARK'}
            />

            <div className="summary-split-grid">
              {/* Champion Architecture Readout (Strictly from /api/models) */}
              <div className="champion-readout">
                <div>
                  <div className="champion-header">
                    <span className="champion-title">{text(championModel, ['name', 'model', 'model_name'])}</span>
                    <span className="champion-badge">{text(championModel, ['type', 'architecture_type'], 'Evaluator')}</span>
                  </div>
                  <p className="champion-desc">
                    {text(championModel, ['description', 'architecture', 'desc'], 'Empirical model metrics evaluated across held-out evaluation timesteps.')}
                  </p>
                </div>

                <div className="champion-metrics-grid">
                  <div className="champion-metric-box highlight">
                    <span className="metric-label">PR-AUC</span>
                    <strong className="metric-val text-cyan">{champPrAuc}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Recall</span>
                    <strong className="metric-val">{champRecall}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Precision</span>
                    <strong className="metric-val">{champPrecision}</strong>
                  </div>
                  <div className="champion-metric-box">
                    <span className="metric-label">Accuracy</span>
                    <strong className="metric-val">{champAccuracy}</strong>
                  </div>
                </div>
              </div>

              {/* Top Flagged Entities (Derived strictly from /api/suspicious) */}
              <div className="summary-queue">
                <div className="queue-header">
                  <span className="queue-title">Top Flagged Test Entities</span>
                  <Label>Ranked by Calibrated Risk</Label>
                </div>

                {!suspiciousRecords.length ? (
                  <Empty message="No suspicious transaction records currently returned by /api/suspicious." />
                ) : (
                  <div className="queue-list">
                    {suspiciousRecords.slice(0, 4).map((record, index) => {
                      const txId = text(record, ['id', 'transaction_id', 'txId'])
                      const cRisk = num(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk'])
                      const prob = num(record, ['probability', 'model_probability'])

                      return (
                        <div className="queue-row" key={txId || index}>
                          <div className="queue-entity">
                            <Target size={16} className="text-coral" />
                            <span>Tx {txId}</span>
                          </div>
                          <div className="queue-stats">
                            <span className="queue-prob font-mono">Prob {prob}</span>
                            <span className="status-pill status-high font-mono">{cRisk} RISK</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}

                <button className="queue-btn" onClick={() => onNavigate('explorer')}>
                  <span>Inspect All {suspiciousRecords.length} Flagged Transactions in Risk Explorer</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}

/* =========================================================================
   View 02: Risk Explorer (100% Dynamic Accountable Table)
   ========================================================================= */
function Explorer({
  rows,
  loading,
  error,
  retry,
  query,
  setQuery,
}: {
  rows: ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
  query: string
  setQuery: (value: string) => void
}) {
  return (
    <div className="workbench-view">
      <div className="view-masthead">
        <Label>Investigation Workbench / 02</Label>
        <h1>Risk Explorer</h1>
        <p>Dense, accountable review of empirical model predictions across the held-out test transactions.</p>
      </div>

      <Card className="workbench-card">
        <div className="workbench-toolbar">
          <div className="toolbar-stats">
            <Label>Test Period Suspicious Records</Label>
            <h2>{loading ? 'Synchronizing backend rankings...' : `${rows.length} visible test records`}</h2>
          </div>
          <div className="table-search-bar">
            <Search size={18} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by Transaction ID or Address..."
            />
            {query && (
              <button className="clear-search-btn" onClick={() => setQuery('')}>
                Clear
              </button>
            )}
          </div>
        </div>

        {loading ? (
          <Loading />
        ) : error ? (
          <ErrorState error={error} retry={retry} />
        ) : !rows.length ? (
          <Empty message="No matching test transactions found for this query." />
        ) : (
          <div className="table-wrap">
            <table className="dense-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Transaction ID</th>
                  <th>Timestep</th>
                  <th>Model Probability</th>
                  <th>Calibrated Risk</th>
                  <th>Model Architecture</th>
                  <th>Total Degree</th>
                  <th>Risk Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((record, index) => {
                  const cRisk = numberValue(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk']) ?? 0
                  const isHigh = cRisk >= 0.5
                  const isElevated = cRisk >= 0.2 && cRisk < 0.5
                  const statusClass = isHigh ? 'status-high' : isElevated ? 'status-elevated' : 'status-monitored'

                  return (
                    <tr key={String(pick(record, ['id', 'transaction_id', 'txId']) ?? index)}>
                      <td className="font-mono text-muted font-bold">#{text(record, ['rank']) || index + 1}</td>
                      <td className="font-mono text-cyan font-bold">{text(record, ['id', 'transaction_id', 'txId'])}</td>
                      <td className="font-mono">{text(record, ['timestep', 'Time step'])}</td>
                      <td className="font-mono">{num(record, ['probability', 'model_probability'])}</td>
                      <td className={`font-mono font-bold ${isHigh ? 'text-coral' : isElevated ? 'text-amber' : 'text-cyan'}`}>
                        {num(record, ['calibrated_risk', 'calibratedRisk', 'calibrated_illicit_risk'])}
                      </td>
                      <td>
                        <span className="model-chip">{text(record, ['model', 'model_name'])}</span>
                      </td>
                      <td className="font-mono">{text(record, ['total_degree'])}</td>
                      <td>
                        <span className={`status-pill ${statusClass}`}>{text(record, ['status', 'label'])}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

/* =========================================================================
   View 03: Model Lab (100% Dynamic Model Cards from /api/models)
   ========================================================================= */
function ModelLab({
  data,
  loading,
  error,
  retry,
}: {
  data?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const rows = Array.isArray((data as any)?.models) ? (data as any).models : records(data)

  return (
    <div className="studio-view">
      <div className="view-masthead">
        <Label>Evaluation Studio / 03</Label>
        <h1>Model Lab</h1>
        <p>
          Empirical held-out benchmark results on Timesteps 35–49. No fabricated confusion matrices or speculative metrics.
        </p>
      </div>

      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorState error={error} retry={retry} />
      ) : !rows.length ? (
        <Empty message="Model metrics unavailable until /api/models returns data." />
      ) : (
        <div className="model-studio-grid">
          {rows.map((record: ApiRecord, index: number) => {
            const prAuc = num(record, ['pr_auc', 'benchmark_pr_auc', 'auc', 'roc_auc'])
            const prec = num(record, ['precision'])
            const rec = num(record, ['recall'])
            const f1 = num(record, ['f1', 'f1_score'])
            const isChampion = record.id === 'risk_scorer'

            return (
              <Card key={index} className={`model-tile-card ${isChampion ? 'champion-card' : ''}`}>
                <div className="tile-top">
                  <div className="tile-index">0{index + 1}</div>
                  <div className="tile-badge">{text(record, ['type', 'architecture_type'])}</div>
                </div>

                <div className="tile-header">
                  <div className="tile-title">{text(record, ['name', 'model', 'model_name'])}</div>
                  <p className="tile-desc">{text(record, ['description', 'desc', 'architecture'])}</p>
                </div>

                <div className="metric-quartet-grid">
                  <div className="metric-box">
                    <span className="metric-box-label">Precision</span>
                    <strong className="metric-box-value">{prec}</strong>
                  </div>
                  <div className="metric-box">
                    <span className="metric-box-label">Recall</span>
                    <strong className="metric-box-value">{rec}</strong>
                  </div>
                  <div className="metric-box">
                    <span className="metric-box-label">F1 Score</span>
                    <strong className="metric-box-value">{f1}</strong>
                  </div>
                  <div className="metric-box highlight-box">
                    <span className="metric-box-label">PR-AUC</span>
                    <strong className="metric-box-value text-cyan">{prAuc}</strong>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}

function EvaluationMetrics({ data, modelsData, loading, error, retry }: {
  data?: ApiRecord
  modelsData?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const groups = (data?.groups || {}) as Record<string, Array<ApiRecord>>
  const models = Array.isArray(data?.models) ? data.models as ApiRecord[] : []
  const modelCards = records(modelsData)
  const modelCardById = new Map(modelCards.map((model) => [String(model.id), model]))
  const beforeComparison = (data?.before_preprocessing || {}) as ApiRecord
  const beforeModels = (beforeComparison.models || {}) as Record<string, ApiRecord>
  const headings: Record<string, string> = { regression: 'Regression metrics', classification: 'Classification metrics' }
  const matrixModels = preprocessingAlgorithms
  const matrixMetrics = preprocessingMetrics
  const savedClassification = groups.classification || []
  const savedValue = (metricId: string, modelId: string): number | null => {
    const row = savedClassification.find((item) => String(item.metric) === metricId)
    const results = (row?.results || {}) as Record<string, ApiRecord>
    const value = results[modelId]?.value
    return typeof value === 'number' && Number.isFinite(value) ? value : null
  }
  const savedRegressionValue = (metricId: string, modelId: string): ApiRecord | undefined => {
    return savedRegressionMetric(data, metricId, modelId)
  }
  const performanceFor = (modelId: string) => {
    const model = modelCardById.get(modelId)
    const confusion = (model?.confusion_matrix || {}) as ApiRecord
    const tn = numberValue(confusion, ['tn'])
    const fp = numberValue(confusion, ['fp'])
    const fn = numberValue(confusion, ['fn'])
    const tp = numberValue(confusion, ['tp'])
    const total = tn !== null && fp !== null && fn !== null && tp !== null ? tn + fp + fn + tp : 0
    const ratio = (numerator: number, denominator: number) => denominator > 0 ? numerator / denominator : null
    const accuracy = total > 0 ? ratio(tn! + tp!, total) : numberValue(model, ['accuracy'])
    const precision = total > 0 ? ratio(tp!, tp! + fp!) : numberValue(model, ['precision'])
    const recall = total > 0 ? ratio(tp!, tp! + fn!) : numberValue(model, ['recall'])
    const specificity = total > 0 ? ratio(tn!, tn! + fp!) : null
    const f1 = precision !== null && recall !== null ? ratio(2 * precision * recall, precision + recall) : numberValue(model, ['f1', 'f1_score'])
    const cardMetricValues = {
      accuracy: numberValue(model, ['accuracy']),
      precision: numberValue(model, ['precision']),
      recall: numberValue(model, ['recall']),
      f1_score: numberValue(model, ['f1', 'f1_score']),
      pr_auc: numberValue(model, ['pr_auc']),
    }
    const savedMetricValues = {
      accuracy: savedValue('accuracy', modelId),
      precision: savedValue('precision', modelId),
      recall: savedValue('recall', modelId),
      f1_score: savedValue('f1_score', modelId),
    }
    const savedMatchesCard = (Object.keys(savedMetricValues) as Array<keyof typeof savedMetricValues>).every((key) => {
      const cardValue = cardMetricValues[key]
      const saved = savedMetricValues[key]
      return cardValue !== null && saved !== null && Math.abs(cardValue - saved) <= 0.00005
    }) && (() => {
      const reference = (model?.benchmark_reference || {}) as ApiRecord
      const referenceMatrix = (reference.confusion_matrix || {}) as ApiRecord
      const rTn = numberValue(referenceMatrix, ['tn'])
      const rFp = numberValue(referenceMatrix, ['fp'])
      const rFn = numberValue(referenceMatrix, ['fn'])
      const rTp = numberValue(referenceMatrix, ['tp'])
      if (rTn === null || rFp === null || rFn === null || rTp === null) return false
      const referenceAccuracy = ratio(rTn + rTp, rTn + rFp + rFn + rTp)
      const referenceValues = {
        accuracy: referenceAccuracy,
        precision: numberValue(reference, ['precision']),
        recall: numberValue(reference, ['recall']),
        f1_score: numberValue(reference, ['f1']),
        pr_auc: numberValue(reference, ['pr_auc']),
      }
      return (Object.keys(cardMetricValues) as Array<keyof typeof cardMetricValues>).every((key) => {
        const cardValue = cardMetricValues[key]
        const referenceValue = referenceValues[key]
        return cardValue !== null && referenceValue !== null && Math.abs(cardValue - referenceValue) <= 0.00005
      })
    })()
    const savedRocAuc = savedValue('auc_roc', modelId)
    return {
      accuracy,
      precision,
      recall,
      specificity,
      f1,
      pr_auc: cardMetricValues.pr_auc,
      roc_auc: numberValue(model, ['roc_auc']) ?? (savedMatchesCard ? savedRocAuc : null),
      fpr: total > 0 ? ratio(fp!, fp! + tn!) : null,
      fnr: total > 0 ? ratio(fn!, fn! + tp!) : null,
      error_rate: accuracy !== null ? 1 - accuracy : null,
      roc_auc_note: savedMatchesCard ? '' : 'Saved ROC-AUC does not match this model card result set.',
    } as Record<string, number | null | string>
  }
  return (
    <div className="studio-view">
      <div className="view-masthead">
        <Label>Evaluation Results / 05</Label>
        <h1>Metrics from saved results</h1>
        <p>Values are read from the evaluation module’s CSV exports for the held-out labeled test set. Undefined and unavailable values remain explicit.</p>
      </div>
      {loading ? <Loading /> : error ? <ErrorState error={error} retry={retry} /> : data?.status !== 'available' ? <Empty message={String(data?.reason || 'Evaluation results are unavailable. Run the evaluation module to create its CSV exports.')} /> : (
        <div className="evaluation-sections">
          {(['classification', 'regression'] as const).map((task) => {
            const rows = groups[task] || []
            return <Card key={task} className="evaluation-card">
              <SectionHead eyebrow="Saved CSV results" title={headings[task]} meta={`${rows.length} metrics`} />
              {!rows.length ? <Empty message={`No ${task} metric rows are available.`} /> : <div className="table-wrap evaluation-table-wrap"><table className="dense-table evaluation-table">
                <thead><tr><th>Metric</th>{models.map((model) => <th key={String(model.id)}>{String(model.name)}</th>)}</tr></thead>
                <tbody>{rows.map((row) => {
                  const results = (row.results || {}) as Record<string, ApiRecord>
                  const first = Object.values(results)[0]
                  const label = String(first?.label || row.metric)
                  return <tr key={String(row.metric)}><td className="metric-name-cell">{label}</td>{models.map((model) => {
                    const result = results[String(model.id)]
                    const status = String(result?.status || 'unavailable')
                    const shown = status === 'calculated' && typeof result?.value === 'number' ? result.value.toFixed(4) : status
                    return <td key={String(model.id)}><span className={`evaluation-value status-${status}`}>{shown}</span></td>
                  })}</tr>
                })}</tbody>
              </table></div>}
            </Card>
          })}
          <Card className="evaluation-card performance-comparison-card">
            <SectionHead
              eyebrow="Matrices"
              title="Preprocessing Impact"
              meta="5 algorithms · 8 metrics"
            />
            <p className="performance-comparison-subtitle">Probability error against the binary illicit label, compared before and after feature preprocessing for the same held-out test transactions.</p>
            <div className="table-wrap evaluation-table-wrap performance-comparison-table-wrap">
              <table className="dense-table performance-comparison-table">
                <thead><tr><th>Algorithm</th><th>Metric</th><th>Before Preprocessing</th><th>After Preprocessing</th><th>Change</th></tr></thead>
                <tbody>{matrixModels.map((algorithm) => {
                  const beforeMetrics = (beforeModels[algorithm.id]?.regression_metrics || {}) as Record<string, ApiRecord>
                  return matrixMetrics.map((metric, index) => {
                    const beforeResult = beforeMetrics[metric.id]
                    const afterResult = savedRegressionValue(metric.id, algorithm.id)
                    const beforeValue = numberValue(beforeResult, ['value'])
                    const afterValue = numberValue(afterResult, ['value'])
                    const beforeStatus = String(beforeResult?.status || 'unavailable')
                    const afterStatus = String(afterResult?.status || 'unavailable')
                    const change = beforeValue !== null && afterValue !== null
                      ? (metric.closerToZero ? Math.abs(beforeValue) - Math.abs(afterValue) : metric.higherIsBetter ? afterValue - beforeValue : beforeValue - afterValue)
                      : null
                    const changeStatus = change === null ? 'unavailable' : Math.abs(change) < 0.00005 ? 'neutral' : change > 0 ? 'improved' : 'deteriorated'
                    const displayValue = (value: number | null, status: string) => status === 'undefined' ? 'Undefined' : status === 'calculated' && value !== null ? value.toFixed(4) : 'N/A'
                    return <tr key={`${algorithm.id}-${metric.id}`}>
                      {index === 0 && <th scope="rowgroup" rowSpan={matrixMetrics.length} className="performance-algorithm-name">{algorithm.name}</th>}
                      <th scope="row" className="performance-metric-name">{metric.label}</th>
                      <td className="performance-before-value">
                        {displayValue(beforeValue, beforeStatus) === 'N/A' ? <span className="performance-unavailable" title={String(beforeComparison.reason || 'Before-preprocessing regression results are unavailable.')}>N/A</span> : displayValue(beforeValue, beforeStatus)}
                      </td>
                      <td className="performance-after-value">
                        {displayValue(afterValue, afterStatus) === 'N/A' ? <span className="performance-unavailable">N/A</span> : displayValue(afterValue, afterStatus)}
                      </td>
                      <td className={`performance-change-value change-${changeStatus}`}>
                        {change === null ? <span className="performance-unavailable">N/A</span> : <span title={metric.closerToZero ? 'Change = |Before| − |After| (closer to zero is better).' : metric.higherIsBetter ? 'Change = After − Before (higher is better).' : 'Change = Before − After (lower is better).'}>
                          {changeStatus === 'improved' ? '↑ ' : changeStatus === 'deteriorated' ? '↓ ' : '• '}{change > 0 ? '+' : ''}{change.toFixed(4)}
                        </span>}
                      </td>
                    </tr>
                  })
                })}</tbody>
              </table>
            </div>
            <p className="evaluation-source performance-source-note">
              Scores compare predicted illicit probabilities with binary test labels, not continuous transaction amounts. Before values are calculated from the saved original-scale experiment predictions; After values come from the saved regression results. Both use labeled timesteps 35–49 (16,670 records). MAPE is undefined because the target includes zeros. Bias / ME improvement means the signed error moves closer to zero. GRU epoch selection and Platt calibration both use validation timesteps 31–34, so calibration-period reuse remains a limitation.
            </p>
          </Card>
          <p className="evaluation-source">Source: <code>{String(data?.source || 'evaluation_metrics/results/*.csv')}</code>. These probability-versus-label regression measures are descriptive; they are not separate continuous-target tasks.</p>
        </div>
      )}
    </div>
  )
}

function PreprocessingErrorAnalysis({ data }: { data?: ApiRecord }) {
  const before = (data?.before_preprocessing || {}) as ApiRecord
  const beforeModels = (before.models || {}) as Record<string, ApiRecord>
  // Each chart uses the exact same API values as the preprocessing comparison matrix.
  const valuesFor = (metricId: string) => preprocessingAlgorithms.map((algorithm) => {
    const beforeMetrics = (beforeModels[algorithm.id]?.regression_metrics || {}) as Record<string, ApiRecord>
    return {
      algorithm: algorithm.chartName,
      fullName: algorithm.name,
      before: comparisonMetricValue(beforeMetrics[metricId]),
      after: comparisonMetricValue(savedRegressionMetric(data, metricId, algorithm.id)),
    }
  })

  return <section className="preprocessing-error-section" aria-labelledby="preprocessing-error-heading">
    <div className="view-masthead preprocessing-error-heading">
      <Label>Preprocessing comparison</Label>
      <h2 id="preprocessing-error-heading">Preprocessing Error Analysis</h2>
      <p>Comparative analysis of prediction error and preprocessing performance across five algorithms, before and after feature preprocessing.</p>
    </div>
    {before.status !== 'available' || data?.status !== 'available' ? (
      <Empty message="Saved before-and-after regression values are unavailable for these charts." />
    ) : <div className="preprocessing-error-grid">
      {preprocessingErrorCharts.map((metric) => {
        const chartData = valuesFor(metric.id)
        const numericValues = chartData.flatMap((row) => [row.before, row.after]).filter((value): value is number => value !== null)
        const min = numericValues.length ? Math.min(...numericValues) : 0
        const max = numericValues.length ? Math.max(...numericValues) : 1
        const span = Math.max(max - min, Math.abs(min) * 0.1, Math.abs(max) * 0.1, 0.1)
        const axisDomain: [number | 'auto', number | 'auto'] = metric.id === 'r2'
          ? [min - span * 0.08, max + span * 0.08]
          : metric.id === 'bias'
            ? [Math.min(0, min) - span * 0.08, Math.max(0, max) + span * 0.08]
            : [0, 'auto']
        const explanation = metric.id === 'bias'
          ? 'Signed mean error; values closer to zero indicate lower systematic prediction bias.'
          : metric.id === 'r2'
            ? 'Higher values indicate better explanatory fit; negative values are retained.'
            : 'Lower values indicate lower prediction error.'
        return <Card className="preprocessing-error-card" key={metric.id}>
          <h3>{metric.title}</h3>
          <p>{explanation}</p>
          <div className="preprocessing-error-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 12, bottom: 28, left: 4 }} barCategoryGap="22%">
                <CartesianGrid stroke="#e5e0d3" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="algorithm" interval={0} angle={-24} textAnchor="end" height={58} tickMargin={8} tick={{ fill: '#56594f', fontSize: 11 }} axisLine={{ stroke: '#cfc8b8' }} tickLine={false} />
                <YAxis domain={axisDomain} tick={{ fill: '#65675d', fontSize: 11 }} tickFormatter={(value) => Number(value).toFixed(2)} axisLine={false} tickLine={false} width={58} label={{ value: metric.axisLabel, angle: -90, position: 'insideLeft', fill: '#56594f', fontSize: 11 }} />
                <Tooltip
                  formatter={(value, name) => [typeof value === 'number' ? value.toFixed(4) : 'N/A', name]}
                  labelFormatter={(_label, payload) => String(payload?.[0]?.payload?.fullName || _label)}
                  contentStyle={{ background: '#fffefa', border: '1px solid #e0dacb', borderRadius: 8, color: '#242820' }}
                  itemStyle={{ color: '#242820', fontSize: 12 }}
                />
                <Legend itemSorter={(item) => item.dataKey === 'before' ? 0 : 1} wrapperStyle={{ color: '#42463b', fontSize: 12, paddingTop: 8 }} />
                {(metric.id === 'r2' || metric.id === 'bias') && <ReferenceLine y={0} stroke="#a95543" strokeWidth={2} />}
                <Bar dataKey="before" name="Before Preprocessing" fill={violet} radius={[3, 3, 0, 0]} />
                <Bar dataKey="after" name="After Preprocessing" fill={cyan} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      })}
    </div>}
  </section>
}

function GraphGallery({ data, evaluationData, loading, error, retry }: {
  data?: ApiRecord
  evaluationData?: ApiRecord
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const graphRows = Array.isArray(data?.graphs) ? data.graphs as ApiRecord[] : []
  const skippedGraphs = Array.isArray(data?.skipped_graphs) ? data.skipped_graphs as ApiRecord[] : []
  const groups = ['regression', 'classification']
  return <div className="studio-view">
    <div className="view-masthead">
      <Label>Generated Figures / 06</Label>
      <h1>Metric graph gallery</h1>
      <p>Rendered PNG figures from the Python visualization scripts. Missing figures are omitted and reported by the generation manifest.</p>
    </div>
    {loading ? <Loading /> : error ? <ErrorState error={error} retry={retry} /> : !graphRows.length ? <Empty message={String(data?.reason || 'No generated metric graphs are available. Run the documented graph generator after evaluation results exist.')} /> : groups.map((task) => {
      const rows = graphRows.filter((item) => item.task === task)
      if (!rows.length) return null
      return <section className="gallery-group" key={task}>
        <SectionHead eyebrow="Python-generated PNG" title={`${task[0].toUpperCase()}${task.slice(1)} visualizations`} meta={`${rows.length} figures`} />
        <div className="metric-gallery-grid">{rows.map((item) => {
          const metric = String(item.metric)
          const title = metric.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
          return <Card className="metric-figure-card" key={`${task}-${metric}`}>
            <h3>{title}</h3>
            <p>Compared across saved model results · {Number(item.dpi) || 320} DPI</p>
            <a href={String(item.url)} target="_blank" rel="noreferrer" aria-label={`Open ${title} graph in a new tab`}>
              <img
                src={String(item.url)}
                alt={`${title} comparison chart for available models`}
                loading="lazy"
                onError={(event) => {
                  const link = event.currentTarget.parentElement
                  if (link) link.textContent = 'This generated figure is currently unavailable.'
                }}
              />
            </a>
          </Card>
        })}</div>
      </section>
    })}
    {skippedGraphs.map((item) => <p className="evaluation-source" key={`skipped-${String(item.metric)}`}>
      {String(item.metric).toUpperCase()} graph skipped: {String(item.reason || 'No calculated result was available.')}
    </p>)}
    <PreprocessingErrorAnalysis data={evaluationData} />
  </div>
}

/* =========================================================================
   View 04: Live Inference Console (Dynamic Presets from /api/suspicious)
   ========================================================================= */
function Inference({
  score,
  suspiciousData,
  loading,
  error,
  retry,
}: {
  score?: ApiRecord | ApiRecord[]
  suspiciousData?: ApiRecord | ApiRecord[]
  loading: boolean
  error?: Error
  retry: () => void
}) {
  const row = records(score)[0]
  const [selectedModel, setSelectedModel] = useState<'gcn' | 'graphsage' | 'gru' | 'logistic'>('gcn')
  const [txInput, setTxInput] = useState('')
  const [scoring, setScoring] = useState(false)
  const [customScore, setCustomScore] = useState<ApiRecord | null>(null)

  // Derive dynamic quick presets from real suspicious data
  const suspiciousRows = records(suspiciousData)
  const dynamicPresets = useMemo(() => {
    return suspiciousRows.slice(0, 3).map((item) => String(pick(item, ['id', 'transaction_id', 'txId'])))
  }, [suspiciousRows])

  // Automatically default input to top real entity once loaded
  useEffect(() => {
    if (!txInput && dynamicPresets.length > 0) {
      setTxInput(dynamicPresets[0])
    }
  }, [dynamicPresets, txInput])

  const activeRecord = customScore || row

  const handleScore = async () => {
    if (!txInput.trim()) return
    setScoring(true)
    try {
      const res = await fetchBackend<ApiRecord>('/api/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tx_id: parseInt(txInput, 10), model: selectedModel }),
      })
      setCustomScore(res)
    } catch (e) {
      console.error(e)
    } finally {
      setScoring(false)
    }
  }

  const cRisk = numberValue(activeRecord, ['calibrated_risk', 'calibratedRisk', 'risk_score']) ?? 0
  const prob = numberValue(activeRecord, ['probability', 'model_probability']) ?? 0
  const isHigh = cRisk >= 0.5
  const isElevated = cRisk >= 0.2 && cRisk < 0.5

  return (
    <div className="inference-view">
      <div className="view-masthead">
        <Label>Scoring Console / 04</Label>
        <h1>Live Inference</h1>
        <p>On-demand evaluation across the held-out test transactions with separate model probabilities and calibrated risk.</p>
      </div>

      <div className="inference-split-grid">
        <Card className="inference-input-card">
          <SectionHead eyebrow="Inference Parameters" title="Query Specification" meta="DUAL-STACK API" />

          <div className="inference-form">
            <div className="form-group">
              <label>Model Architecture</label>
              <div className="model-selector-tabs">
                {[
                  { id: 'gcn', label: 'GCN' },
                  { id: 'graphsage', label: 'GraphSAGE' },
                  { id: 'gru', label: 'Temporal GRU' },
                  { id: 'logistic', label: 'Logistic' },
                ].map(({ id, label }) => (
                  <button
                    key={id}
                    type="button"
                    className={`model-tab-btn ${selectedModel === id ? 'active' : ''}`}
                    onClick={() => setSelectedModel(id as any)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Transaction ID (Held-Out Test Split)</label>
              <div className="tx-input-wrap">
                <input
                  type="text"
                  value={txInput}
                  onChange={(e) => setTxInput(e.target.value)}
                  placeholder="Enter Transaction ID..."
                />
                <button
                  type="button"
                  className="score-submit-btn"
                  onClick={handleScore}
                  disabled={scoring || !txInput.trim()}
                >
                  {scoring ? <RefreshCw size={16} className="spin" /> : <Zap size={16} />}
                  <span>{scoring ? 'Evaluating...' : 'Score Entity'}</span>
                </button>
              </div>
            </div>

            <div className="preset-quick-select">
              <span>Dynamic Quick Presets:</span>
              {!dynamicPresets.length ? (
                <span className="text-muted">Loading from API...</span>
              ) : (
                dynamicPresets.map((id, idx) => (
                  <button key={id} type="button" onClick={() => setTxInput(id)}>
                    Rank #{idx + 1} ({id})
                  </button>
                ))
              )}
            </div>
          </div>
        </Card>

        <Card className="inference-result-card">
          <SectionHead eyebrow="Verified Artifact Bridge" title="Evaluation Output" meta="/api/score" />

          {loading ? (
            <Loading />
          ) : error ? (
            <ErrorState error={error} retry={retry} />
          ) : !activeRecord ? (
            <Empty message="Inference unavailable until /api/score returns a response." />
          ) : (
            <div className="score-readout-wrap">
              <div className="score-primary-gauges">
                <div className="gauge-box">
                  <span className="gauge-label">Raw Model Probability</span>
                  <strong className="gauge-value text-cyan">{prob.toFixed(4)}</strong>
                  <span className="gauge-sub">Continuous classification signal</span>
                </div>

                <div className="gauge-box">
                  <span className="gauge-label">Calibrated Illicit Risk</span>
                  <strong className={`gauge-value ${isHigh ? 'text-coral' : isElevated ? 'text-amber' : 'text-cyan'}`}>
                    {cRisk.toFixed(4)}
                  </strong>
                  <span className="gauge-sub">Validation-calibrated risk</span>
                </div>
              </div>

              <div className="inference-summary-strip">
                <div className="strip-item">
                  <span>Evaluated Entity</span>
                  <strong className="font-mono text-cyan">Tx {text(activeRecord, ['id', 'transaction_id', 'txId'])}</strong>
                </div>
                <div className="strip-item">
                  <span>Model Selector</span>
                  <strong>{text(activeRecord, ['model', 'requested_model'])}</strong>
                </div>
                <div className="strip-item">
                  <span>Risk Classification</span>
                  <span className={`status-pill ${isHigh ? 'status-high' : isElevated ? 'status-elevated' : 'status-monitored'}`}>
                    {isHigh ? 'HIGH RISK' : isElevated ? 'ELEVATED RISK' : 'MONITORED'}
                  </span>
                </div>
              </div>

              <div className="notice-box">
                <ShieldCheck size={18} className="text-cyan" />
                <p>Model outputs are empirical evidentiary signals, not definitive proof of financial crime.</p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

/* =========================================================================
   Primary Root Shell & Dock Navigation
   ========================================================================= */
export default function IntelligenceDashboard() {
  const [page, setPage] = useState<ViewId>('overview')
  const [query, setQuery] = useState('')

  useEffect(() => {
    const syncRoute = () => {
      const candidate = window.location.hash.slice(1) as ViewId
      setPage(nav.some((item) => item.id === candidate) ? candidate : 'overview')
    }
    syncRoute()
    window.addEventListener('hashchange', syncRoute)
    return () => window.removeEventListener('hashchange', syncRoute)
  }, [])

  const navigate = (view: ViewId) => {
    const nextHash = view === 'overview' ? '' : `#${view}`
    if (window.location.hash !== nextHash) window.location.hash = nextHash
    setPage(view)
  }

  const overview = useSWR('/api/overview', (path) => fetchBackend<ApiRecord>(path))
  const suspicious = useSWR('/api/suspicious', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const models = useSWR('/api/models', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const score = useSWR('/api/score', (path) => fetchBackend<ApiRecord | ApiRecord[]>(path))
  const evaluation = useSWR('/api/evaluation', (path) => fetchBackend<ApiRecord>(path))
  const graphs = useSWR('/api/graphs', (path) => fetchBackend<ApiRecord>(path))

  const isConnected = Boolean(overview.data || models.data || suspicious.data)

  const filteredRows = useMemo(
    () =>
      records(suspicious.data).filter((record) =>
        text(record, ['id', 'transaction_id', 'address', 'hash', 'txId'])
          .toLowerCase()
          .includes(query.toLowerCase())
      ),
    [suspicious.data, query]
  )

  const retry = () => [overview, suspicious, models, score, evaluation, graphs].forEach((feed: Feed) => feed.mutate())

  return (
    <div className="command-shell">
      {/* Top Header Bar */}
      <header className="brand-strip">
        <div className="brand-lockup">
          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>
              elliptic<span>++</span>
            </strong>
            <small>AI FINANCIAL INTELLIGENCE</small>
          </div>
        </div>

        <div className="top-command-controls">
          <div className="brand-meta">
            <span className="live-dot" style={{ background: isConnected ? cyan : amber }} />
            <span>{isConnected ? 'BACKEND CONNECTED · LOCAL API' : 'CONNECTING TO BACKEND...'}</span>
          </div>

          <button className="sync-btn" onClick={retry}>
            <RefreshCw size={14} />
            <span>Sync feeds</span>
          </button>
        </div>
      </header>

      {/* Main Responsive Content Surface (Uses 96-98% viewport width) */}
      <main className="command-main">
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            {page === 'overview' && (
              <Overview
                overviewData={overview.data}
                modelsData={models.data}
                suspiciousData={suspicious.data}
                loading={overview.isLoading || models.isLoading}
                error={overview.error}
                retry={retry}
                onNavigate={navigate}
              />
            )}
            {page === 'explorer' && (
              <Explorer
                rows={filteredRows}
                loading={suspicious.isLoading}
                error={suspicious.error}
                retry={retry}
                query={query}
                setQuery={setQuery}
              />
            )}
            {page === 'models' && (
              <ModelLab data={models.data} loading={models.isLoading} error={models.error} retry={retry} />
            )}
            {page === 'inference' && (
              <Inference
                score={score.data}
                suspiciousData={suspicious.data}
                loading={score.isLoading}
                error={score.error}
                retry={retry}
              />
            )}
            {page === 'metrics' && <EvaluationMetrics data={evaluation.data} modelsData={models.data} loading={evaluation.isLoading} error={evaluation.error} retry={retry} />}
            {page === 'graphs' && <GraphGallery data={graphs.data} evaluationData={evaluation.data} loading={graphs.isLoading} error={graphs.error} retry={retry} />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Page-footer navigation follows the selected view content. */}
      <nav className="floating-dock" aria-label="Primary navigation">
        {nav.map(({ id, label, icon: Icon, index }) => (
          <button
            key={id}
            className={`dock-btn ${page === id ? 'active' : ''}`}
            onClick={() => navigate(id)}
            aria-label={label}
            aria-current={page === id ? 'page' : undefined}
          >
            <span className="dock-index">{index}</span>
            <Icon size={18} />
            <span className="dock-label">{label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}
