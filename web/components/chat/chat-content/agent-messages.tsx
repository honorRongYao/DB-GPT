import ModelIcon from '@/new-components/chat/content/ModelIcon';
import { SwapRightOutlined } from '@ant-design/icons';
import { GPTVis } from '@antv/gpt-vis';
import ReferencesContent from './ReferencesContent';
import markdownComponents, { markdownPlugins, preprocessLaTeX } from './config';

interface Props {
  data: {
    sender: string;
    receiver: string;
    model: string | null;
    markdown: string;
    resource: any;
    // 后端已按国内时间（UTC+8）格式化好的字符串，可能为空
    created_at?: string | null;
  }[];
}

function AgentMessages({ data }: Props) {
  if (!data || !data.length) return null;
  return (
    <>
      {data.map((item, index) => (
        <div key={index} className='rounded'>
          <div className='flex items-center mb-3 text-sm'>
            {item.model ? <ModelIcon model={item.model} /> : <div className='rounded-full w-6 h-6 bg-gray-100' />}
            <div className='ml-2 opacity-70'>
              {item.sender}
              <SwapRightOutlined className='mx-2 text-base' />
              {item.receiver}
            </div>
            {item.created_at && <div className='ml-auto text-xs opacity-50'>{item.created_at}</div>}
          </div>
          <div className='whitespace-normal text-sm mb-3'>
            <GPTVis components={markdownComponents} {...markdownPlugins}>
              {preprocessLaTeX(item.markdown)}
            </GPTVis>
          </div>
          {item.resource && item.resource !== 'null' && <ReferencesContent references={item.resource} />}
        </div>
      ))}
    </>
  );
}

export default AgentMessages;
