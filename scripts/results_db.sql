drop table if exists experiment_result;
drop table if exists experiment_description;
drop table if exists databases;
drop sequence if exists experiment_description_seq;
drop sequence if exists database_idd_seq;


create sequence experiment_description_seq;
create sequence database_idd_seq;

create table experiment_description(idexp integer primary key default nextval('experiment_description_seq'), description text, gspan_variant varchar(100), date timestamp);


create table databases(iddb integer primary key default nextval('database_idd_seq'), name varchar(100));

create table experiment_result(iddb integer references databases(iddb),
       idexp integer references experiment_description(idexp),
       support integer,
       execution_time double precision,
       primary key(iddb, idexp,support));

 
create or replace function get_speedup(expid1 integer, expid2 integer, iddb integer)
       RETURNS TABLE (iddb integer, support integer, idexp1 integer, execution_time1 double precision, idexp2 integer, execution_time2 double precision, speedup double precision) as $$
select *,er2.execution_time/er1.execution_time as diff from experiment_result as er1 join experiment_result as er2 using (iddb,support) where er1.idexp=$1 and er2.idexp=$2 and er1.iddb=$3;
$$ LANGUAGE SQL;

create or replace function get_expname(expid integer)
       returns varchar(100) as $$
select gspan_variant from experiment_description where idexp=$1;
$$ LANGUAGE SQL;

create or replace function get_dbname(dbid integer) returns varchar(100) as $$
select name from databases where iddb=$1;
$$ LANGUAGE SQL;

create or replace function get_speedup_symbolic(expid1 integer, expid2 integer, iddb integer)
 returns table(iddb varchar(100), support integer, idexp1 varchar(100), idexp2 varchar(100), execution_time1 double precision, execution_time2 double precision, speedup double precision) as $$
select get_dbname(sp.iddb), support, get_expname(sp.idexp1), get_expname(sp.idexp2), execution_time1, execution_time2, speedup from get_speedup($1, $2, $3) as sp;
$$ LANGUAGE SQL;


