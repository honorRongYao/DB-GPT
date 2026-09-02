import { AutoChart, BackEndChartType, getChartType } from '@/components/chart/autoChart';
import { formatSql } from '@/utils';
import { Datum } from '@antv/ava';
import { Table, Tabs, TabsProps } from 'antd';
import { CodePreview } from './code-preview';

function ChartView({
  data,
  type,
  sql,
  describe,
}: {
  data: Datum[];
  type: BackEndChartType;
  sql: string;
  describe?: string;
}) {
  const columns = data?.[0]
    ? Object.keys(data?.[0])?.map(item => {
        return {
          title: item,
          dataIndex: item,
          key: item,
        };
      })
    : [];
  const ChartItem = {
    key: 'chart',
    label: 'Chart',
    children: <AutoChart data={data} chartType={getChartType(type)} />,
  };
  const SqlItem = {
    key: 'sql',
    label: 'SQL',
    children: <CodePreview language='sql' code={formatSql(sql ?? '', 'mysql') as string} />,
  };
  const DataItem = {
    key: 'data',
    label: 'Data',
    children: <Table dataSource={data} columns={columns} scroll={{ x: 'auto' }} />,
  };
  const TabItems: TabsProps['items'] = [ChartItem, SqlItem, DataItem];

  return (
    <div>
      {describe ? (
        <div className='mb-2 whitespace-pre-wrap rounded-md bg-theme-light p-3 text-sm leading-7 text-gray-600 dark:bg-theme-dark dark:text-gray-300'>
          {describe}
        </div>
      ) : null}
      <Tabs defaultActiveKey={type === 'response_table' ? 'data' : 'chart'} items={TabItems} size='small' />
    </div>
  );
}

export default ChartView;
