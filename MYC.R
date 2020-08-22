library(TxDb.Hsapiens.UCSC.hg19.knownGene)
txdb <-TxDb.Hsapiens.UCSC.hg19.knownGene
txdb
columns(txdb)        # 查看txdb的columns
keytypes(txdb)       # 查看txdb的keytypes

gene = genes(txdb)   # 获取全部基因的信息

symbal_keys <- c("MYC","MYCL","MYCN")

# 利用org.Hs.eg.db查找基因对应的ENTREZID
ibrary(org.Hs.eg.db)
geneIDselect <- select(org.Hs.eg.db, keys=gene$gene_id,columns=c("SYMBOL","GENENAME"),keytype="ENTREZID" )
geneIDselect <- as.data.frame(geneIDselect)  # 转换为data frame
geneIDselect <- geneIDselect[which(geneIDselect$SYMBOL %in% symbal_keys),] # 选择
geneIDselect

keys <- c("4609", "4610", "4613")

# 查看3个基因的信息
gene <- as.data.frame(gene)  # 将GRanges object转换为data frame
gene <- gene[which(gene$gene_id %in% keys),]    # 选择
gene


# 方法一：直接从txdb中提取
columns <- c("TXID", "TXCHROM", "TXNAME", "TXSTART","TXEND")
# 按照GENEID选择
TC_select1 <- select(txdb, keys = keys, columns=columns, keytype="GENEID")
TC_select1$TXRANGE <- TC_select1$TXEND-TC_select1$TXSTART+1  # 计算长度
TC_select1 <- TC_select1[c("GENEID", "TXID", "TXNAME", "TXRANGE")]
TC_select1


# 方法二： 先获取基因全部转录本再提取
TransGroup <- transcriptsBy(txdb)
# TransGroup数据类型为GRangesList 
TC_select2 <- transcripts(txdb, filter=list(gene_id=keys))
TC_select2 <- as.data.frame(TC_select2)
TC_select2 <- TC_select2[c("tx_id", "tx_name", "width")]
TC_select2
